/*
 * Copyright 2021 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/context/DistributedContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <runtime/local/datastructures/AllocationDescriptorGRPC.h>
#include <runtime/local/io/DaphneSerializer.h>
#include <runtime/distributed/proto/DistributedGRPCCaller.h>
#include <runtime/distributed/proto/worker.pb.h>
#include <runtime/distributed/proto/worker.grpc.pb.h>

#ifdef USE_MPI
    #include <runtime/distributed/worker/MPIHelper.h>
#endif

#include <cassert>
#include <cstddef>


// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<ALLOCATION_TYPE AT, class DT>
struct DistributedCollect {
    static void apply(DT *&mat, DCTX(dctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<ALLOCATION_TYPE AT, class DT>
void distributedCollect(DT *&mat, DCTX(dctx))
{
    DistributedCollect<AT, DT>::apply(mat, dctx);
}



// ****************************************************************************
// (Partial) template specializations for different distributed backends
// ****************************************************************************


// ----------------------------------------------------------------------------
// MPI
// ----------------------------------------------------------------------------
#ifdef USE_MPI
template<class DT>
struct DistributedCollect<ALLOCATION_TYPE::DIST_MPI, DT>
{
    static void apply(DT *&mat, DCTX(dctx)) 
    {
        assert (mat != nullptr && "result matrix must be already allocated by wrapper since only there exists information regarding size");        
        size_t worldSize = MPIHelper::getCommSize();
        for(size_t rank=0; rank<worldSize ; rank++) 
        {
            if(rank==COORDINATOR) // we currently exclude the coordinator
               continue;
            
            std::string address = std::to_string(rank);  
            auto dp=mat->getMetaDataObject()->getDataPlacementByLocation(address);   
            auto distributedData = dynamic_cast<AllocationDescriptorMPI&>(*(dp->allocation)).getDistributedData();            
            WorkerImpl::StoredInfo info = {
                distributedData.identifier,
                distributedData.numRows,
                distributedData.numCols
            };
            MPIHelper::requestData(rank, info);
        }
        auto collectedDataItems = 0u;
        for (size_t i = 1; i < worldSize; i++) {
            size_t len;
            int rank;
            std::vector<char> buffer;
            MPIHelper::getMessage(&rank, TypesOfMessages::OUTPUT, MPI_UNSIGNED_CHAR, buffer, &len);
            
            std::string address = std::to_string(rank);  
            auto dp = mat->getMetaDataObject()->getDataPlacementByLocation(address);   
                    
            auto denseMat = dynamic_cast<DenseMatrix<double>*>(mat);
            if (!denseMat){
                throw std::runtime_error("Distribute grpc only supports DenseMatrix<double> for now");
            }            

            auto slicedMat = dynamic_cast<DenseMatrix<double>*>(DF_deserialize(buffer));
            auto resValues = denseMat->getValues() + (dp->range->r_start * denseMat->getRowSkip());
            auto slicedMatValues = slicedMat->getValues();
            for (size_t r = 0; r < dp->range->r_len; r++) {
                memcpy(resValues + dp->range->c_start, slicedMatValues, dp->range->c_len * sizeof(double));
                resValues += denseMat->getRowSkip();
                slicedMatValues += slicedMat->getRowSkip();
            }
            
            collectedDataItems+=  dp->range->r_len *  dp->range->c_len;

            auto distributedData = dynamic_cast<AllocationDescriptorMPI&>(*(dp->allocation)).getDistributedData();            
            distributedData.isPlacedAtWorker = false;
            dynamic_cast<AllocationDescriptorMPI&>(*(dp->allocation)).updateDistributedData(distributedData);
            // this is to handle the case when not all workers participate in the computation, i.e., number of workers is larger than of the work items
            if(collectedDataItems == denseMat->getNumRows() * denseMat->getNumCols())
                break;
        }
    };
};
#endif

// ----------------------------------------------------------------------------
// Asynchronous GRPC
// ----------------------------------------------------------------------------

template<class DT>
struct DistributedCollect<ALLOCATION_TYPE::DIST_GRPC_ASYNC, DT>
{
    static void apply(DT *&mat, DCTX(dctx)) 
    {
        assert (mat != nullptr && "result matrix must be already allocated by wrapper since only there exists information regarding size");        

        struct StoredInfo{
            size_t dp_id;
        };
        DistributedGRPCCaller<StoredInfo, distributed::StoredData, distributed::Data> caller(dctx);


        auto dpVector = mat->getMetaDataObject()->getDataPlacementByType(ALLOCATION_TYPE::DIST_GRPC);
        for (auto &dp : *dpVector) {
            auto address = dp->allocation->getLocation();
            
            auto distributedData = dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).getDistributedData();
            StoredInfo storedInfo({dp->dp_id});
            distributed::StoredData protoData;
            protoData.set_identifier(distributedData.identifier);
            protoData.set_num_rows(distributedData.numRows);
            protoData.set_num_cols(distributedData.numCols);                       

            caller.asyncTransferCall(address, storedInfo, protoData);
        }
                
        

        while (!caller.isQueueEmpty()){
            auto response = caller.getNextResult();
            auto dp_id = response.storedInfo.dp_id;
            auto dp = mat->getMetaDataObject()->getDataPlacementByID(dp_id);
            auto data = dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).getDistributedData();            

            auto matProto = response.result;
            
            // TODO: We need to handle different data types 
            auto denseMat = dynamic_cast<DenseMatrix<double>*>(mat);
            if (!denseMat){
                throw std::runtime_error("Distribute grpc only supports DenseMatrix<double> for now");
            }
            // Zero copy buffer
            std::vector<char> buf(static_cast<const char*>(matProto.bytes().data()), static_cast<const char*>(matProto.bytes().data()) + matProto.bytes().size()); 
            auto slicedMat = dynamic_cast<DenseMatrix<double>*>(DF_deserialize(buf));
            auto resValues = denseMat->getValues() + (dp->range->r_start * denseMat->getRowSkip());
            auto slicedMatValues = slicedMat->getValues();
            for (size_t r = 0; r < dp->range->r_len; r++){
                memcpy(resValues + dp->range->c_start, slicedMatValues, dp->range->c_len * sizeof(double));
                resValues += denseMat->getRowSkip();                    
                slicedMatValues += slicedMat->getRowSkip();
            }               
            data.isPlacedAtWorker = false;
            dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).updateDistributedData(data);
        } 
    };
};


// ----------------------------------------------------------------------------
// Synchronous GRPC
// ----------------------------------------------------------------------------

// template<class DT>
// struct DistributedCollect<ALLOCATION_TYPE::DIST_GRPC_SYNC, DT>
// {
//     static void apply(DT *&mat, DCTX(dctx)) 
//     {
//         assert (mat != nullptr && "result matrix must be already allocated by wrapper since only there exists information regarding size");        

//         auto ctx = DistributedContext::get(dctx);
//         std::vector<std::thread> threads_vector;

//         auto dpVector = mat->getMetaDataObject()->getDataPlacementByType(ALLOCATION_TYPE::DIST_GRPC);
//         for (auto &dp : *dpVector) {
//             auto address = dp->allocation->getLocation();
            
//             auto distributedData = dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).getDistributedData();            
//             distributed::StoredData protoData;
//             protoData.set_identifier(distributedData.identifier);
//             protoData.set_num_rows(distributedData.numRows);
//             protoData.set_num_cols(distributedData.numCols);                       

//             std::thread t([address, dp = dp.get(), protoData, distributedData, &mat, &ctx]() mutable
//             {
//                 auto stub = ctx->stubs[address].get();

//                 distributed::Data matProto;
//                 grpc::ClientContext grpc_ctx;
//                 stub->Transfer(&grpc_ctx, protoData, &matProto);
            
//                 // TODO: We need to handle different data types 
//                 auto denseMat = dynamic_cast<DenseMatrix<double>*>(mat);
//                 if (!denseMat){
//                     throw std::runtime_error("Distribute grpc only supports DenseMatrix<double> for now");
//                 }
//                 // Zero copy buffer
//                 std::vector<char> buf(static_cast<const char*>(matProto.bytes().data()), static_cast<const char*>(matProto.bytes().data()) + matProto.bytes().size()); 
//                 auto slicedMat = dynamic_cast<DenseMatrix<double>*>(DF_deserialize(buf));
//                 auto resValues = denseMat->getValues() + (dp->range->r_start * denseMat->getRowSkip());
//                 auto slicedMatValues = slicedMat->getValues();
//                 for (size_t r = 0; r < dp->range->r_len; r++){
//                     memcpy(resValues + dp->range->c_start, slicedMatValues, dp->range->c_len * sizeof(double));
//                     resValues += denseMat->getRowSkip();                    
//                     slicedMatValues += slicedMat->getRowSkip();
//                 }               
//                 distributedData.isPlacedAtWorker = false;
//                 dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).updateDistributedData(distributedData);
//             });
//             threads_vector.push_back(move(t));        
//         }
//         for (auto &thread : threads_vector)
//             thread.join();
//     };
// };


template<class DT>
struct DistributedCollect<ALLOCATION_TYPE::DIST_GRPC_SYNC, DT>
{
    static void apply(DT *&mat, DCTX(dctx)) 
    {
        assert(mat != nullptr && "Result matrix must be allocated by the wrapper since only there exists information regarding size.");        

        auto ctx = DistributedContext::get(dctx);
        std::vector<std::thread> threads_vector;

        auto dpVector = mat->getMetaDataObject()->getDataPlacementByType(ALLOCATION_TYPE::DIST_GRPC);
        for (auto &dp : *dpVector) {
            auto address = dp->allocation->getLocation();
            
            auto distributedData = dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).getDistributedData();            
            distributed::StoredData protoData;
            protoData.set_identifier(distributedData.identifier);
            protoData.set_num_rows(distributedData.numRows);
            protoData.set_num_cols(distributedData.numCols);

        
            
            std::thread t([address, dp = dp.get(), protoData, distributedData, &mat, &ctx]() mutable
            {
                auto stub = ctx->stubs[address].get();

                // Request the data from the worker in chunks
                // std::unique_ptr<grpc::ClientReader<distributed::Data>> reader=stub->Transfer(&grpc_ctx, protoData));
                
                // Initialize variables for receiving and storing the chunks
                Structure* mat = nullptr; // Replace 'Structure' with the appropriate data type
                distributed::Data matProto;
                auto reader=stub->Transfer(&grpc_ctx, protoData, &matProto);
                reader->Read(&matProto);
                auto buffer = matProto.bytes().data();
                auto len = matProto.bytes().size();

                auto denseMat = dynamic_cast<DenseMatrix<double>*>(mat);
                if (!denseMat){
                    throw std::runtime_error("Distribute grpc only supports DenseMatrix<double> for now");
                }
                    
                // Handle single value case
                if (DF_Dtype(buffer) == DF_data_t::Value_t) {
                    std::vector<char> buf(static_cast<const char*>(matProto.bytes().data()), static_cast<const char*>(matProto.bytes().data()) + matProto.bytes().size()); 
                    auto slicedMat = dynamic_cast<DenseMatrix<double>*>(DF_deserialize(buf));
                    auto resValues = denseMat->getValues() + (dp->range->r_start * denseMat->getRowSkip());
                    auto slicedMatValues = slicedMat->getValues();
                    for (size_t r = 0; r < dp->range->r_len; r++){
                        memcpy(resValues + dp->range->c_start, slicedMatValues, dp->range->c_len * sizeof(double));
                        resValues += denseMat->getRowSkip();                    
                        slicedMatValues += slicedMat->getRowSkip();
                    }               
                    distributedData.isPlacedAtWorker = false;
                    dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).updateDistributedData(distributedData);
                } else {  //handle chunks
                    // Initialize deserializer for chunks
                    deserializer.reset(new DaphneDeserializerChunks<Structure>(&mat, len));
                    deserializerIter.reset(new DaphneDeserializerChunks<Structure>::Iterator(deserializer->begin()));
                
                
                    // Store the chunks
                    (*deserializerIter)->second->resize(len);
                    (*deserializerIter)->first = len;
                    
                    if ((*deserializerIter)->second->size() < len)
                        (*deserializerIter)->second->resize(len);
                    (*deserializerIter)->second->assign(static_cast<const char*>(buffer), static_cast<const char*>(buffer) + len);
                    
                    // Advance the iterator, this partially deserializes
                    ++(*deserializerIter);

                    // Need changes for chunks
                    // for (size_t r = 0; r < dp->range->r_len; r++){
                    //     memcpy(resValues + dp->range->c_start, slicedMatValues, dp->range->c_len * sizeof(double));
                    //     resValues += denseMat->getRowSkip();                    
                    //     slicedMatValues += slicedMat->getRowSkip();
                    // }               
                    // distributedData.isPlacedAtWorker = false;
                    // dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).updateDistributedData(distributedData);
                    
                    while (reader->Read(&data)){
                        (*deserializerIter)->first = len;
                        if ((*deserializerIter)->second->size() < len)
                            (*deserializerIter)->second->resize(len);
                        (*deserializerIter)->second->assign(static_cast<const char*>(buffer), static_cast<const char*>(buffer) + len);
                        
                        // advance iterator, this also partially deserializes
                        ++(*deserializerIter);

                        // Need changes for chunks
                        // for (size_t r = 0; r < dp->range->r_len; r++){
                        //     memcpy(resValues + dp->range->c_start, slicedMatValues, dp->range->c_len * sizeof(double));
                        //     resValues += denseMat->getRowSkip();                    
                        //     slicedMatValues += slicedMat->getRowSkip();
                        // }               
                        // distributedData.isPlacedAtWorker = false;
                        // dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).updateDistributedData(distributedData);
                        
                    }
                }
            });
            threads_vector.push_back(std::move(t));
        }
        for (auto &thread : threads_vector)
            thread.join();
    };
};


template<class DT>
struct DistributedCollect<ALLOCATION_TYPE::DIST_GRPC_SYNC, DT>
{
    static void apply(DT *&mat, DCTX(dctx)) 
    {
        assert(mat != nullptr && "Result matrix must be allocated by the wrapper since only there exists information regarding size.");        

        auto ctx = DistributedContext::get(dctx);
        std::vector<std::thread> threads_vector;

        auto dpVector = mat->getMetaDataObject()->getDataPlacementByType(ALLOCATION_TYPE::DIST_GRPC);
        for (auto &dp : *dpVector) {
            auto address = dp->allocation->getLocation();
            
            auto distributedData = dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).getDistributedData();            
            distributed::StoredData protoData;
            protoData.set_identifier(distributedData.identifier);
            protoData.set_num_rows(distributedData.numRows);
            protoData.set_num_cols(distributedData.numCols);

            std::thread t([address, dp = dp.get(), protoData, distributedData, &mat, &ctx]() mutable
            {
                auto stub = ctx->stubs[address].get();

                // Request the data from the worker in chunks
                grpc::ClientContext grpc_ctx;
                std::unique_ptr<grpc::ClientReader<distributed::Data>> reader(stub->Transfer(&grpc_ctx, protoData));

                // Initialize variables for receiving and storing the chunks
                Structure* mat = nullptr; // Replace 'Structure' with the appropriate data type

                // Initialize deserializer for chunks
                std::unique_ptr<DaphneDeserializerChunks<Structure>> deserializer;
                std::unique_ptr<typename DaphneDeserializerChunks<Structure>::Iterator> deserializerIter;

                while (reader->Read(&data)) {
                    auto buffer = data.bytes().data();
                    auto len = data.bytes().size();

                    // Handle single value case
                    if (DF_Dtype(buffer) == DF_data_t::Value_t) {
                        // Deserialization of single value
                        std::vector<char> buf(static_cast<const char*>(data.bytes().data()), static_cast<const char*>(data.bytes().data()) + data.bytes().size()); 
                        auto slicedMat = dynamic_cast<DenseMatrix<double>*>(DF_deserialize(buf));

                        auto resValues = denseMat->getValues() + (dp->range->r_start * denseMat->getRowSkip());
                        auto slicedMatValues = slicedMat->getValues();

                        for (size_t r = 0; r < dp->range->r_len; r++){
                            memcpy(resValues + dp->range->c_start, slicedMatValues, dp->range->c_len * sizeof(double));
                            resValues += denseMat->getRowSkip();                    
                            slicedMatValues += slicedMat->getRowSkip();
                        }               

                        // Update data placement info
                        distributedData.isPlacedAtWorker = false;
                        dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).updateDistributedData(distributedData);
                    } else {
                        // Initialize deserializer for chunks
                        if (!deserializer) {
                            deserializer.reset(new DaphneDeserializerChunks<Structure>(&mat, len));
                            deserializerIter.reset(new typename DaphneDeserializerChunks<Structure>::Iterator(deserializer->begin()));
                        }
                    
                        // Store the chunks
                        (*deserializerIter)->second->resize(len);
                        (*deserializerIter)->first = len;
                        
                        if ((*deserializerIter)->second->size() < len)
                            (*deserializerIter)->second->resize(len);
                        (*deserializerIter)->second->assign(static_cast<const char*>(buffer), static_cast<const char*>(buffer) + len);
                        
                        // Advance the iterator, this partially deserializes
                        ++(*deserializerIter);
                    }
                }
            });
            threads_vector.push_back(std::move(t));
        }
        for (auto &thread : threads_vector)
            thread.join();
    };
};

template<class DT>
struct DistributedCollect<ALLOCATION_TYPE::DIST_GRPC_SYNC, DT>
{
    static void apply(DT *&mat, DCTX(dctx)) 
    {
        // Assert that the matrix is allocated by the wrapper
        assert(mat != nullptr && "Result matrix must be allocated by the wrapper since only there exists information regarding size.");        

        // Get the distributed context
        auto ctx = DistributedContext::get(dctx);
        std::vector<std::thread> threads_vector;

        // Get data placements for distributed GRPC
        auto dpVector = mat->getMetaDataObject()->getDataPlacementByType(ALLOCATION_TYPE::DIST_GRPC);
        for (auto &dp : *dpVector) {
            auto address = dp->allocation->getLocation();
            
            // Get distributed data information
            auto distributedData = dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).getDistributedData();            
            distributed::StoredData protoData;
            protoData.set_identifier(distributedData.identifier);
            protoData.set_num_rows(distributedData.numRows);
            protoData.set_num_cols(distributedData.numCols);

            // Thread to handle transfer and deserialization
            std::thread t([address, dp = dp.get(), protoData, distributedData, &mat, &ctx]() mutable {
                auto stub = ctx->stubs[address].get();

                Structure* temp_mat = nullptr;

                auto reader = stub->Transfer(&grpc_ctx, protoData);

                std::vector<std::pair<size_t, std::shared_ptr<std::vector<char>>>> receivedChunks;

                distributed::Data data;
                while (reader->Read(&data)) {
                    auto buffer = data.bytes().data();
                    auto len = data.bytes().size();

                    // Check the data type
                    if (DF_Dtype(buffer) == DF_data_t::Value_t) {
                        throw std::runtime_error("Invalid data type.");
                    } else {
                        if (temp_mat == nullptr) {
                            // Replace 'Structure' constructor with your appropriate constructor parameters
                            temp_mat = new Structure(distributedData.numRows, distributedData.numCols);
                        }

                        // Store received chunks in a vector
                        receivedChunks.emplace_back(len, std::make_shared<std::vector<char>>(buffer, buffer + len));
                    }
                }

                if (temp_mat != nullptr) {
                    // Initialize deserializer
                    DaphneDeserializerChunks<Structure> deserializer(&temp_mat, temp_mat->getNumRows() * temp_mat->getNumCols());
                    DaphneDeserializerChunks<Structure>::Iterator deserializerIter = deserializer.begin();

                    for (const auto &chunk : receivedChunks) {
                        // Assign received chunks to deserializer iterator buffer
                        (*deserializerIter)->second->resize(chunk.first);
                        (*deserializerIter)->second->assign(chunk.second->begin(), chunk.second->end());

                        // Advance iterator to deserialize partially
                        ++deserializerIter;
                    }

                    // Lock the matrix and update
                    std::lock_guard<std::mutex> lock(mat->getMutex());

                    for (size_t i = 0; i < temp_mat.getNumRows(); ++i) {
                        for (size_t j = 0; j < temp_mat.getNumCols(); ++j) {
                            mat[i][j] = temp_mat.getValue(i, j);
                        }
                    }
                    mat->update(temp_mat);
                    delete temp_mat;
                }
            });

            threads_vector.push_back(std::move(t));
        }

        // Join the threads
        for (auto &thread : threads_vector)
            thread.join();
    }
};



// data placements
// Structure parameters
// distributedData
//testcommit