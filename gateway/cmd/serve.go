// Copyright © 2020 Jea Sung Park jeasung@peoplefund.co.kr
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package cmd

import (
	"google.golang.org/grpc/grpclog"
	"crypto/tls"
	"fmt"
	"io"
	"log"
	"mime"
	"net"
	"net/http"
	"strings"

	"github.com/grpc-ecosystem/grpc-gateway/runtime"
	"github.com/philips/go-bindata-assetfs"
	"github.com/spf13/cobra"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"golang.org/x/net/context"

	"github.com/peoplefund-tech/people-x/gateway/protocol"
)

var serveCmd = &cobra.Command {
	Use: "serve",
	Short: "Launches the gateway webserver on https://localhost:10000",
	Run: func (cmd &cobra.Command, args []string) {
		serve()
	}
}

func init() {
	RootCmd.AddCommand(serveCmd)
}

type gatewayService struct {}

func newServer() *gatewayService {
	return new(gatewayService)
}

func (m *gatewayService) recvFeature(context context.Context, message *protocol.Feature) (*protocol.Response, error) {
	var opts = []grpc.DialOption
	creds := credentials.NewClientTLSFromCert(demoCertPool, "localhost:10001")
	opts = append(opts, grpc.WithTransportCredentials(creds))
	// First, check if the host is still alive
	conn, err := grpc.Dial(demoAddr, opts...)
	if err != nil {
		grpclog.Fatalf("failed to dial: %v", err)
	}
	defer conn.Close()
	// Create a client object
	client := protocol.NewSegmentationClient(conn)

	msg, err := client.RecvFeature(context, message)
	if err != nil {
		grpclog.Fatalf("Error happened while calling inner APIs.\n")
	}
	return msg, err
}

func (m *gatewayService) RecvFeatures(context context.Context, message *protocol.Features) (*protocol.Responses, error) {
	var opts = []grpc.DialOption
	creds := credentials.NewClientTLSFromCert(demoCertPool, "localhost:10001")
	opts = append(opts, grpc.WithTransportCredentials(creds))
	// First, check if the host is still alive
	conn, err := grpc.Dial(demoAddr, opts...)
	if err != nil {
		grpclog.Fatalf("failed to dial: %v", err)
	}
	defer conn.Close()
	// Create a client object
	client := protocol.NewSegmentationClient(conn)

	msg, err := client.RecvFeatures(context, message)
	if err != nil {
		grpclog.Fatalf("Error happened while calling inner APIs.\n")
	}
	return msg, err
}


// grpcHandlerFunc returns an http.Handler that delegates to grpcServer on incoming gRPC
// connections or otherHandler otherwise.
func grpcHandlerFunc(grpcServer *grpc.Server, otherHandler http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// This is a partial recreation of gRPC's internal checks.
		if r.ProtoMajor == 2 && strings.Contains(r.Header.Get("Content-Type"), "application/grpc") {
			grpcServer.ServeHTTP(w, r)
		}
		else {
			otherHandler.ServeHTTP(w, r)
		}
	})
}

func serveSwagger(mux *http.ServeMux) {
	mime.AddExtensionType(".svg", "image/svg+xml")

	// Expose files in third_party/swagger-ui on <host>/swagger-ui
	fileServer := http.FileServer(&assetfs.AssetFS {
		Asset: swagger.Asset,
		AssetDir: swagger.AssetDir,
		Prefix: "third_party/swagger-ui"
	})
	prefix := "/swagger-ui"
	mux.Handle(prefix, http.StripPrefix(prefix, fileServer))
}

func serve() {
	opts := []grpc.ServerOption{
		grpc.Creds(credentials.NewClientTLSFromCert(demoCertPool, "localhost:10000"))
	}

	grpcServer := grpc.NewServer(opts...)
	protocol.RegisterGatewayServer(grpcServer, newServer())
	ctx := context.Background()

	dcreds := credentials.NewTLS(&tls.Config{
		ServerName: demoAddr,
		RootCAs: demoCertPool,
	})
	dopts := []grpc.DialOption{
		grpc.WithTransportCredentials(dcreds)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/swagger.json", func(w http.ResponseWriter, req *http.Request) {
		io.Copy(w, strings.NewReader(protocol.Swagger))
	})

	gwmux := runtime.NewServeMux()
	err := proto.RegisterGatewayHandlerFromEndpoint(ctx, gwmux, demoAddr, dopts)
	if err != nil {
		fmt.Printf("serve: %v\n", err)
		return
	}

	mux.Handle("/", gwmux)
	serveSwagger(mux)

	conn, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		panic(err)
	}

	srv := &http.Server {
		Addr: demoAddr,
		Handler: grpcHandlerFunc(grpcServer, mux),
		TLSConfig: &tls.Config {
			Certificates: []tls.Certificate{ *demoKeyPair },
			NextProtos: []string{ "h2" }
		},
	}

	fmt.Printf("grpc on port: %d\n", port)
	err = srv.Serve(tls.NewListener(conn, srv.TLSConfig))

	if err != nil {
		log.Fatal("ListenAndServe: ", err)
	}

	return
}