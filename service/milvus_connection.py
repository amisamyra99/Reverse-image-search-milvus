"""Creation of a singleton Connection """

from pymilvus import MilvusClient

# Initialize and connect to Milvus
#milvus_client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")

milvus_client = MilvusClient(uri="https://in03-9274f3f30a5d7b9.serverless.gcp-us-west1.cloud.zilliz.com", token="0579cf6a0d68f692ecabcb0ecf6353f8881aa24e8e1bfe1542c82c8fccaf5b81bb887722ae46ff1a3510104f5810802dddf4048c")

