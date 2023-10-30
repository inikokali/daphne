/*
 *  Copyright 2021 The DAPHNE Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */


#include "WorkerImplGRPCSync.h"

#include <runtime/local/io/DaphneSerializer.h>
#include <runtime/local/datastructures/DataObjectFactory.h>

#include <grpcpp/grpcpp.h>
#include <grpcpp/server_builder.h>

WorkerImplGRPCSync::WorkerImplGRPCSync(const std::string& addr, DaphneUserConfig& _cfg) : WorkerImpl(_cfg)

{
    builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
    builder.RegisterService(this);
    builder.SetMaxReceiveMessageSize(INT_MAX);
    builder.SetMaxSendMessageSize(INT_MAX);
    server = builder.BuildAndStart();
}

void WorkerImplGRPCSync::Wait() {
    server->Wait();
}

grpc::Status WorkerImplGRPCSync::Store(::grpc::ServerContext *context,
                         ::grpc::ServerReader< ::distributed::Data>* reader,
                         ::distributed::StoredData *response) 
{
    StoredInfo storedInfo;
    distributed::Data data;
    reader->Read(&data);

    auto buffer = data.bytes().data();
    auto len = data.bytes().size();
    if (DF_Dtype(buffer) == DF_data_t::Value_t) {
        double val = DaphneSerializer<double>::deserialize(buffer);
        storedInfo = WorkerImpl::Store(&val);
        
        response->set_identifier(storedInfo.identifier);
        response->set_num_rows(storedInfo.numRows);
        response->set_num_cols(storedInfo.numCols);
    } else {
        deserializer.reset(new DaphneDeserializerChunks<Structure>(&mat, len));
        deserializerIter.reset(new DaphneDeserializerChunks<Structure>::Iterator(deserializer->begin()));    

        (*deserializerIter)->second->resize(len);
        (*deserializerIter)->first = len;
        
        if ((*deserializerIter)->second->size() < len)
            (*deserializerIter)->second->resize(len);
        (*deserializerIter)->second->assign(static_cast<const char*>(buffer), static_cast<const char*>(buffer) + len);
        
        // advance iterator, this also partially deserializes
        ++(*deserializerIter);
        while (reader->Read(&data)){
            buffer = data.bytes().data();
            len = data.bytes().size();
            (*deserializerIter)->first = len;
            if ((*deserializerIter)->second->size() < len)
                (*deserializerIter)->second->resize(len);
            (*deserializerIter)->second->assign(static_cast<const char*>(buffer), static_cast<const char*>(buffer) + len);
            
            // advance iterator, this also partially deserializes
            ++(*deserializerIter);
        }
        storedInfo = WorkerImpl::Store(mat);
        response->set_identifier(storedInfo.identifier);
        response->set_num_rows(storedInfo.numRows);
        response->set_num_cols(storedInfo.numCols);
    }
    return ::grpc::Status::OK;
}

grpc::Status WorkerImplGRPCSync::Compute(::grpc::ServerContext *context,
                         const ::distributed::Task *request,
                         ::distributed::ComputeResult *response)
{
    std::vector<StoredInfo> inputs;
    inputs.reserve(request->inputs().size());

    std::vector<StoredInfo> outputs = std::vector<StoredInfo>();
    for (auto input : request->inputs()){
        auto stored = input.stored();
        inputs.push_back(StoredInfo({stored.identifier(), stored.num_rows(), stored.num_cols()}));
    }
    auto respMsg = WorkerImpl::Compute(&outputs, inputs, request->mlir_code());
    for (auto output : outputs){        
        distributed::WorkData workData;        
        workData.mutable_stored()->set_identifier(output.identifier);
        workData.mutable_stored()->set_num_rows(output.numRows);
        workData.mutable_stored()->set_num_cols(output.numCols);
        *response->add_outputs() = workData;
    }
    if (respMsg.ok())
        return ::grpc::Status::OK;
    else
        return ::grpc::Status(grpc::StatusCode::ABORTED, respMsg.error_message());        
}

grpc::Status WorkerImplGRPCSync::Transfer(::grpc::ServerContext *context,
                          const ::distributed::StoredData *request,
                         ::distributed::Data *response)
{
    StoredInfo info({request->identifier(), request->num_rows(), request->num_cols()});
    std::vector<char> buffer;
    size_t bufferLength;
    Structure *mat = WorkerImpl::Transfer(info);
    bufferLength = DaphneSerializer<Structure>::serialize(mat, buffer);
    response->set_bytes(buffer.data(), bufferLength);
    return ::grpc::Status::OK;
}


grpc::Status WorkerImplGRPCSync::Transfer(::grpc::ServerContext* context,
                                          const ::distributed::StoredData* request,
                                          ::grpc::ServerWriter< ::distributed::Data>* writer)
{
    StoredInfo info({request->identifier(), request->num_rows(), request->num_cols()});
    std::vector<char> buffer;
    size_t bufferLength;
    Structure *mat = WorkerImpl::Transfer(info);
    bufferLength = DaphneSerializer<Structure>::serialize(mat, buffer);
    response->set_bytes(buffer.data(), bufferLength);
    return ::grpc::Status::OK;
}



grpc::Status WorkerImplGRPCSync::Transfer(::grpc::ServerContext* context,
                                          const ::distributed::StoredData* request,
                                          ::grpc::ServerWriter< ::distributed::Data>* writer) {
    StoredInfo info({request->identifier(), request->num_rows(), request->num_cols()});
    Structure* mat = WorkerImpl::Transfer(info);

    // Perform serialization in chunks
    std::vector<char> buffer;
    size_t bufferLength;
    size_t serializeFromByte = 0;
    const size_t chunkSize = DaphneSerializer<Structure>::DEFAULT_SERIALIZATION_BUFFER_SIZE; // Define your chunk size

    do {
        bufferLength = DaphneSerializer<Structure>::serialize(mat, buffer, chunkSize, serializeFromByte);
        
        // Send the serialized chunk of data
        ::distributed::Data response;
        response.set_bytes(buffer.data(), bufferLength);
        writer->Write(response);

        serializeFromByte += bufferLength;
    } while (bufferLength > 0);

    return ::grpc::Status::OK;
}

grpc::Status WorkerImplGRPCSync::Transfer(::grpc::ServerContext* context,
                          const ::distributed::StoredData* request,
                          ::grpc::ServerWriter< ::distributed::Data>* writer)
{
    StoredInfo info({request->identifier(), request->num_rows(), request->num_cols()});
    Structure* mat = WorkerImpl::Transfer(info);

    // TODO: Determine an appropriate chunk size.
    // For instance, let's use a random chunk size (for instance, 100 elements).
    const int chunkSize = 4096;

    // Serialize the 'mat' structure to be sent in chunks.
    std::vector<char> buffer;
    auto serializer = DaphneSerializerChunks<DT>(mat, chunkSize);


    for (size_t i = 0; i < serializer.size(); i += chunkSize) {
        size_t remaining = std::min(chunkSize, serializer.size() - i);
        buffer.clear();
        serializer.serializeChunk(i, remaining, buffer);  // Serialize a chunk of data.

        // Prepare and send the chunk to the coordinator.
        ::distributed::Data data;
        data.set_bytes(buffer.data(), buffer.size());
        writer->Write(data);
    }

    // Signal the end of the stream.
    writer->Finish();

    return ::grpc::Status::OK;
}

grpc::Status WorkerImplGRPCSync::Transfer(::grpc::ServerContext *context,
                      const ::distributed::StoredData *request,
                      ::grpc::ServerWriter< ::distributed::Data>* writer)
{
    StoredInfo info({request->identifier(), request->num_rows(), request->num_cols()});
    Structure *mat = WorkerImpl::Transfer(info);

    // Assuming DaphneSerializer<Structure>::serializeChunk() serializes individual chunks.
    // Serialize and send chunks one by one
    for (/* iterate through chunks or individual data elements */) {
        std::vector<char> buffer;
        size_t bufferLength;
        // Serialize the current chunk or data element
        bufferLength = DaphneSerializer<Structure>::serializeChunk(currentChunk, buffer);

        // Prepare the Data message with serialized chunk/data
        ::distributed::Data data;
        data.set_bytes(buffer.data(), bufferLength);

        // Write the data to the stream
        if (!writer->Write(data)) {
            // Handle a failure to write, if required
            return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Failed to send data.");
        }
    }

    // Indicate the completion of sending data
    writer->WritesDone();

    return ::grpc::Status::OK;
}

grpc::Status WorkerImplGRPCSync::Transfer(::grpc::ServerContext* context,
                                         const ::distributed::StoredData* request,
                                         ::grpc::ServerWriter< ::distributed::Data>* writer) {
    StoredInfo info({request->identifier(), request->num_rows(), request->num_cols()});
    Structure* mat = WorkerImpl::Transfer(info);

    // Split the data into chunks and serialize them
    std::vector<char> buffer;
    size_t chunkSize = 2048
    size_t totalSize = DaphneSerializer<Structure>::size(mat);

    for (size_t offset = 0; offset < totalSize; offset += chunkSize) {
        size_t remainingSize = std::min(chunkSize, totalSize - offset);
        buffer.clear();

        // Serialize a chunk of data
        DaphneSerializer<Structure>::serializeChunk(mat, offset, remainingSize, buffer);

        // Create a Data message for the chunk and send it
        distributed::Data data;
        data.set_bytes(buffer.data(), buffer.size());
        writer->Write(data);
    }

    return ::grpc::Status::OK;
}