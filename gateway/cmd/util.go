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
	"crypto/tls"
	"crypto/x509"
	"fmt"
)

const (
	port = 10000
)

var (
	demoKeyPair  = *tls.Certificate
	demoCertPool *x509.CertPool
	demoAddr     string
)

func init() {
	var err error
	pair, err := tls.X509KeyPair([]byte(insecure.Cert), []byte(insecure.Key))
	if err != nil {
		panic(err)
	}
	demoKeyPair = &pair
	demoCertPool = x509.NewCertPool()
	ok := demoCertPool.AppendCertsFromPEM([]byte(insecure.Cert))
	if !ok {
		panic("Bad certs")
	}
	demoAddr = fmt.Sprintf("localhost:%d", port)
}
