from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
from PIL import Image
import os
from service.milvus_connection import milvus_client  # Import the singleton connection
import numpy as np
from service.feature_extractor import FeatureExtractor

# Collection name and vector dimension (ResNet50 model output)
COLLECTION_NAME = "image_vectors"
VECTOR_DIM =512

# Check if the collection exists; if not, create it
def create_collection():
    """Create a collection if it does not exist."""
    if milvus_client.has_collection(collection_name=COLLECTION_NAME):
        milvus_client.drop_collection(collection_name=COLLECTION_NAME)
    milvus_client.create_collection(
        collection_name=COLLECTION_NAME,
        vector_field_name="vector",
        dimension=VECTOR_DIM,
        auto_id=True,
        enable_dynamic_field=True,
        metric_type="COSINE",
    )

create_collection()



def index_images_to_milvus(image_dir: str,  
                            collection_name: str = "image_vectors",
                            model_name: str = "resnet34") -> None:
    """Indexes images from a directory into a Milvus collection.

    Args:
        image_dir (str): The path to the directory containing images.
        client (MilvusClient): An instance of the connected MilvusClient.
        collection_name (str, optional): The name of the Milvus collection. 
                                          Defaults to "image_embeddings".
        model_name (str, optional): The name of the TIMM model for feature extraction. 
                                    Defaults to "resnet34".
    """

    extractor = FeatureExtractor(model_name) 

    for root, _, files in os.walk(image_dir):
        for filename in files:
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                filepath = os.path.join(root, filename)

                image_embedding = extractor(filepath)  # Extract embedding

                milvus_client.insert(
                    collection_name,
                    {"vector": image_embedding, "filename": filepath},
                )

    print(f"Images from '{image_dir}' indexed to Milvus collection '{collection_name}'")

index_images_to_milvus("D:/3third-semester/product_images")

def add_images_to_milvus(image_folder):
    """Add image vectors to Milvus using HTTP."""
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    vectors = []
    feature_extractor = FeatureExtractor(modelname="resnet50")
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path).convert('RGB')
        
        # Extract feature vector using the pre-trained model
        image_vector = feature_extractor(image)
          # Ensure it's a NumPy array of floats
        #image_vector = np.array(image_vector, dtype=np.float32) 
        # Sanity check dimensions
        #if len(image_vector) != VECTOR_DIM:
        #   raise ValueError(f"Unexpected vector length: {len(image_vector)}")
        
        vectors.append(image_vector)

        print(type(vectors[0]))  # Should be <class 'numpy.ndarray'>
        print(vectors[0].dtype) # Should be float32 (or similar float type) 
        print(vectors[0].shape) # Should be (VECTOR_DIM,)
    
    # Insert the data into Milvus - ensure the collection schema is correct
    milvus_client.insert(
        collection_name=COLLECTION_NAME,
        data=vectors,  # Directly pass the list of vectors
    )
    
    print(f"Inserted {len(image_files)} images into the collection.")

def search_image_in_milvus(query_vector, limit=2):
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = milvus_client.search(
        collection_name=COLLECTION_NAME, 
        data=[query_vector], 
        anns_field="image_vectors", 
        param=search_params,  # Pass search parameters here
        limit=limit 
    )
    
    # Extract top results
    output = [{"image_id": hit["entity"]["image_id"], "score": hit["distance"]} for hit in results]
    return output
