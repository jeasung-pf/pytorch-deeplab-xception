package protocol 

const (
gateway.swagger = `{
  "swagger": "2.0",
  "info": {
    "title": "gateway.proto",
    "version": "version not set"
  },
  "consumes": [
    "application/json"
  ],
  "produces": [
    "application/json"
  ],
  "paths": {
    "/v1/multiple/recognition": {
      "post": {
        "operationId": "recvFeatures",
        "responses": {
          "200": {
            "description": "A successful response.",
            "schema": {
              "$ref": "#/definitions/protocolResponses"
            }
          },
          "default": {
            "description": "An unexpected error response",
            "schema": {
              "$ref": "#/definitions/runtimeError"
            }
          }
        },
        "parameters": [
          {
            "name": "body",
            "in": "body",
            "required": true,
            "schema": {
              "$ref": "#/definitions/protocolFeatures"
            }
          }
        ],
        "tags": [
          "Gateway"
        ]
      }
    },
    "/v1/recognition": {
      "post": {
        "operationId": "recvFeature",
        "responses": {
          "200": {
            "description": "A successful response.",
            "schema": {
              "$ref": "#/definitions/protocolResponse"
            }
          },
          "default": {
            "description": "An unexpected error response",
            "schema": {
              "$ref": "#/definitions/runtimeError"
            }
          }
        },
        "parameters": [
          {
            "name": "body",
            "in": "body",
            "required": true,
            "schema": {
              "$ref": "#/definitions/protocolFeature"
            }
          }
        ],
        "tags": [
          "Gateway"
        ]
      }
    }
  },
  "definitions": {
    "protobufAny": {
      "type": "object",
      "properties": {
        "type_url": {
          "type": "string"
        },
        "value": {
          "type": "string",
          "format": "byte"
        }
      }
    },
    "protocolFeature": {
      "type": "object",
      "properties": {
        "image_encoded": {
          "type": "string",
          "format": "byte"
        },
        "image_filename": {
          "type": "string"
        },
        "image_format": {
          "type": "string"
        },
        "image_height": {
          "type": "integer",
          "format": "int32"
        },
        "image_width": {
          "type": "integer",
          "format": "int32"
        },
        "image_segmentation_class_encoded": {
          "type": "string",
          "format": "byte"
        },
        "image_segmentation_class_format": {
          "type": "string"
        }
      }
    },
    "protocolFeatures": {
      "type": "object",
      "properties": {
        "features": {
          "type": "array",
          "items": {
            "$ref": "#/definitions/protocolFeature"
          }
        }
      }
    },
    "protocolResponse": {
      "type": "object",
      "properties": {
        "image_filename": {
          "type": "string"
        },
        "image_recognition_numbers": {
          "type": "string"
        },
        "image_recognition_verification": {
          "type": "string"
        },
        "image_recognition_dates": {
          "type": "string"
        }
      }
    },
    "protocolResponses": {
      "type": "object",
      "properties": {
        "responses": {
          "type": "array",
          "items": {
            "$ref": "#/definitions/protocolResponse"
          }
        }
      }
    },
    "runtimeError": {
      "type": "object",
      "properties": {
        "error": {
          "type": "string"
        },
        "code": {
          "type": "integer",
          "format": "int32"
        },
        "message": {
          "type": "string"
        },
        "details": {
          "type": "array",
          "items": {
            "$ref": "#/definitions/protobufAny"
          }
        }
      }
    }
  }
}
`
)
