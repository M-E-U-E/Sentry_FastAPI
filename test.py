#CRUD

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Sample in-memory database
database = []
id_counter = 1  # Track the next available ID
deleted_ids = []  # Store deleted IDs for reuse

# Pydantic Model (without ID input)
class Item(BaseModel):
    name: str
    price: float
    available: bool = True

# Function to get the next available ID
def get_next_id():
    global id_counter
    if deleted_ids:
        return deleted_ids.pop(0)  # Reuse the lowest deleted ID
    else:
        current_id = id_counter
        id_counter += 1
        return current_id

# Create an item (POST)
@app.post("/items/", response_model=dict)
def create_item(item: Item):
    new_id = get_next_id()  # Get the next available ID
    new_item = {"id": new_id, "name": item.name, "price": item.price, "available": item.available}
    database.append(new_item)
    return new_item

# Read all items (GET)
@app.get("/items/", response_model=List[dict])
def get_items():
    return database

# Read a single item by ID (GET)
@app.get("/items/{item_id}", response_model=dict)
def get_item(item_id: int):
    for item in database:
        if item["id"] == item_id:
            return item
    raise HTTPException(status_code=404, detail="Item not found")

# Update an item (PUT)
@app.put("/items/{item_id}", response_model=dict)
def update_item(item_id: int, updated_item: Item):
    for index, item in enumerate(database):
        if item["id"] == item_id:
            database[index] = {"id": item_id, **updated_item.dict()}
            return database[index]
    raise HTTPException(status_code=404, detail="Item not found")

# Delete an item (DELETE)
@app.delete("/items/{item_id}")
def delete_item(item_id: int):
    global deleted_ids
    for index, item in enumerate(database):
        if item["id"] == item_id:
            del database[index]
            deleted_ids.append(item_id)  # Store the deleted ID for reuse
            deleted_ids.sort()  # Ensure IDs are reused in order
            return {"message": "Item deleted successfully"}
    raise HTTPException(status_code=404, detail="Item not found")