from fastapi import APIRouter, File, UploadFile, HTTPException
from service.milvus_service import add_images_to_milvus, search_image_in_milvus
from service.feature_extractor import FeatureExtractor
from io import BytesIO
from PIL import Image
from service.milvus_connection import milvus_client  # Import the singleton connection
import os
import numpy as np

router = APIRouter()



@router.post("/upload-images/")
async def upload_images(files: list[UploadFile] = File(...)):
    """Endpoint to upload images and add them to the Milvus collection."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    # Temporary directory for storing uploaded images
    temp_dir = "temp_images"
    
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    for file in files:
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

    # Add uploaded images to Milvus
    add_images_to_milvus(temp_dir)

    # Clean up: Remove temporary files
    for file in files:
        os.remove(os.path.join(temp_dir, file.filename))

    return {"status": "Images uploaded and processed successfully"}

@router.post("/search/")
async def search_image(file: UploadFile = File(...)):
    """Endpoint to search for similar images in the Milvus collection."""
    feature_extractor = FeatureExtractor(modelname="resnet50")

    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # Read the uploaded file
    image = Image.open(BytesIO(await file.read())).convert("RGB")

    # Extract feature vector from the uploaded image
    query_vector = feature_extractor(image)

    # Perform the search in Milvus
    results = search_image_in_milvus(query_vector)

    return {"results": results}



@router.get("/collections/")
async def list_collections():
    collections=milvus_client.list_collections()
    return{'collections':collections}


@router.get("/collection-info/")
async def collection_info(collection_name:str):
    if not collection_name:
        HTTPException(status_code=400,detail="Collection name is required")
     # Check if the collection exists
    collections = milvus_client.list_collections()
    if collection_name not in collections:
        raise HTTPException(status_code=404, detail="Collection not found")
    
    # Get the number of entities in the collection
    collection = milvus_client.get_collection_stats(collection_name)
      

    return {"collection_name": collection_name, "stats": collection}

@router.post("/delete/")
async def delete_collection(collection_name:str):
    collections=milvus_client.drop_collection(collection_name=collection_name)
    return{'the name of the collection deleted is ':collections}