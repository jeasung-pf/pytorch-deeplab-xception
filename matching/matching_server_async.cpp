/*
 *
 * Copyright 2015 gRPC authors.
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
 *
 */

#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include <grpcpp/grpcpp.h>
#include <grpc/support/log.h>

#ifdef BAZEL_BUILD
#include "matching.grpc.pb.h"
#else
#include "matching.grpc.pb.h"
#endif

#include <include/Bootstrap.hpp>
#include <include/Classifier.hpp>
#include <include/Matcher.hpp>
#include <params.hpp>

using grpc::Server;
using grpc::ServerAsyncResponseWriter;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerCompletionQueue;
using grpc::Status;

using protocol::Feature;
using protocol::Features;
using protocol::Response;
using protocol::Responses;
using protocol::Matching;

class ServerImpl final {
public:
    ~ServerImpl() {
        server_->Shutdown();
        // Always shutdown the completion queue after the server.
        cq_->Shutdown();
    }

    // There is no shutdown handling in this code.
    void Run() {
        std::string server_address("0.0.0.0:50051");

        ServerBuilder builder;
        // Listen on the given address without any authentication mechanism.
        builder.AddListeningPort(server_address, grpc::InsecureChannelCredentials());
        // Register "service_" as the instance through which we'll communicate with
        // clients. In this case it corresponds to an *asynchronous* service.
        builder.RegisterService(&service_);
        // Get hold of the completion queue used for the asynchronous communication
        // with the gRPC runtime.
        cq_ = builder.AddCompletionQueue();
        // Finally assemble the server.
        server_ = builder.BuildAndStart();
        std::cout << "Server listening on " << server_address << std::endl;

        // Process to the server's main loop.
        HandleRpcs();
    }

private:
    // Class encompassing the state and login needed to serve a request.
    class CallData {
    public:
        // Take in the "service" instance (in this case representing an asynchronous server)
        // and the completion queue "cq" used for asynchronous communication with the gRPC
        // runtime.
        CallData(Matching::AsyncService *service, ServerCompletionQueue *cq) : service_(service), cq_(cq),
                                                                               responder_(&ctx_), status_(CREATE) {
            Proceed();
        }

        void Proceed() {
            if (status_ == CREATE) {
                // Make this instance progress to the PROCESS state.
                status_ = PROCESS;
                // As part of the initial CREATE state, we *request* that the system start
                // processing recvFeature/recvFeatures requests. In this request, "this" acts
                // are the tag uniquely identifying the request (so that different CallData
                // instances can serve different requests concurrently), in this case the
                // memory address of this CallData instance.
                service_->RequestrecvFeature(&ctx_, &feature_, &responder_, cq_, cq_, this);
            } else if (status_ == PROCESS) {
                // Spawn a new CallData instance to serve new clients while we process the one
                // for this CallData. The instance will deallocate itself as part of its
                // FINISH state.
                new CallData(service_, cq_);

                // The actual processing.
                // TODO. implement image processing methods
                std::string image_encoded = feature_.image_encoded();
                std::string image_filename = feature_.image_filename();
                std::string image_segmentation_class_encoded feature_.image_segmentation_class_encoded();
                std::string image_segmentation_class_format = feature_.image_segmentation_class_format();

                cv::Mat image_decoded = cv::imdecode(image_encoded, cv::IMREAD_COLOR);
                cv::Mat image_segmentation_class_decoded = cv::imdecode(image_segmentation_class_encoded, cv::IMREAD_COLOR);

                cv::Ptr<cv::Matcher::FeatureParams> featureParams = cv::Matcher::ORBFeatureParams();
                cv::Ptr<cv::flann::IndexParams> indexParams = cv::flann::LshIndexParams(1, 10, 2);
                cv::Ptr<cv::flann::SearchParams> searchParams = cv::flann::SearchParams();

                cv::Ptr<cv::Matcher::Matcher> model = new cv::Matcher::Matcher(featureParams, indexParams, searchParams);
                cv::Mat transformed;
                model->predict(image_decoded,
                               image_segmentation_class_decoded,
                               transformed,
                               cv::Matcher::identification_t::ID_DRIVERS_LICENSE);
                // And we are done! Let the gRPC runtime know we've finished, using the
                // memory address of this instance as the uniquely identifying tag for
                // the event.
                status_ = FINISH;
                responder_.Finish(response_, Status::OK, this);
            } else {
                GPR_ASSERT(status_ == FINISH);
                delete this;
            }
        }


    private:
        // The means of communication with the gRPC runtime for an asynchronous server.
        Matching::AsyncService *service_;
        // The producer-consumer queue where for asynchronous server notifications.
        ServerCompletionQueue *cq_;
        // Context for the rpc, allowing to tweak aspects of it such as the use of compression,
        // authentication, as well as to send metadata back to the client.
        ServerContext ctx_;
        // What we get from the client;
        Feature feature_;
        Features features_;
        Response response_;
        Responses responses_;
        // The means to get back to the client.
        ServerAsyncResponseWriter<Response> responder_;
        // Let's implement a tiny state machine with the following states.
        enum CallStatus {
            CREATE, PROCESS, FINISH
        };
        CallStatus status_;
    };

    // This can be run in multiple threads if needed.
    void HandleRpcs() {
        // Spawn a new CallData instance to serve new clients.
        new CallData(&service_, cq_.get());
        void *tag;
        bool ok;
        while (true) {
            // Block waiting to read the next event from the completion queue. The event is
            // uniquely identified by its tag, which in this case is the memory address
            // of a CallData instance.
            // The return value of Next should always be checked. This return value tells us
            // whether there is any kind of event or cq_ is shutting down.
            GPR_ASSERT(cq_->Next(&tag, &ok));
            GPR_ASSERT(ok);
            static_cast<CallData*>(tag)->Proceed();
        }
    }

    std::unique_ptr<ServerCompletionQueue> cq_;
    Matching::AsyncService service_;
    std::unique_ptr<Server> server_;
};


int main(int argc, char** argv) {
//    RunServer();

    return 0;
}
