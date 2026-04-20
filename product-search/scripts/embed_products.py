#!/usr/bin/env python3
"""
Pre-compute embeddings for product documents using sentence-transformers.

Usage:
    # Embed an existing JSONL file (adds/overwrites "embedding" field)
    python embed_products.py --input products.jsonl --output products_embedded.jsonl

    # Generate synthetic products with embeddings
    python embed_products.py --generate 10000 --output products_embedded.jsonl

    # Use a different model
    python embed_products.py --input products.jsonl --output out.jsonl --model all-MiniLM-L6-v2

Requirements:
    pip install sentence-transformers
"""

import argparse
import json
import random
import sys
import time

def generate_synthetic(count: int) -> list[dict]:
    """Generate synthetic product documents (mirrors Rust generator)."""
    adjectives = [
        "Premium", "Classic", "Ultra", "Deluxe", "Essential", "Professional",
        "Vintage", "Modern", "Sleek", "Rugged", "Lightweight", "Heavy-Duty",
        "Compact", "Portable", "Luxury", "Budget", "Smart", "Organic",
        "Natural", "Advanced",
    ]
    materials = [
        "Leather", "Cotton", "Steel", "Aluminum", "Wood", "Bamboo", "Plastic",
        "Ceramic", "Glass", "Silicone", "Nylon", "Polyester", "Titanium",
        "Carbon Fiber", "Copper", "Brass", "Rubber", "Wool", "Linen", "Denim",
    ]
    products = [
        "Jacket", "Shirt", "Pants", "Shoes", "Watch", "Bag", "Wallet", "Belt",
        "Hat", "Gloves", "Scarf", "Sunglasses", "Backpack", "Laptop Stand",
        "Phone Case", "Headphones", "Speaker", "Keyboard", "Mouse", "Cable",
        "Charger", "Lamp", "Mug", "Bottle", "Notebook", "Pen",
        "Desk Organizer", "Chair", "Table", "Shelf",
    ]
    colors = [
        "Red", "Blue", "Green", "Black", "White", "Navy", "Gray", "Brown",
        "Beige", "Olive", "Burgundy", "Teal", "Coral", "Gold", "Silver",
    ]
    brands = [
        "NordicCraft", "UrbanEdge", "PeakGear", "EcoLine", "TechVault",
        "SkyBound", "IronForge", "CrystalWave", "ThunderBolt", "SilverStream",
        "GreenHaven", "BlueCrest", "RedFox", "GoldenPath", "OakHeart",
        "StormPeak", "CoralReef", "ArcticWind", "SunForge", "MoonStone",
    ]
    categories = [
        "Electronics", "Clothing", "Accessories", "Home & Garden", "Sports",
        "Office", "Kitchen", "Outdoor", "Travel", "Fitness",
    ]
    sizes = ["XS", "S", "M", "L", "XL", "XXL", "One Size", "Small", "Medium", "Large"]

    docs = []
    for i in range(count):
        adj = random.choice(adjectives)
        mat = random.choice(materials)
        prod = random.choice(products)
        color = random.choice(colors)
        brand = random.choice(brands)
        cat = random.choice(categories)
        size = random.choice(sizes)

        name = f"{adj} {color} {mat} {prod}"
        price = round(random.uniform(5.0, 500.0), 2)
        stock = random.randint(0, 999)
        sku = f"SKU-{i:08d}"

        description = (
            f"The {name} by {brand} is a {adj.lower()} {prod.lower()} made from "
            f"high-quality {mat.lower()}. Available in {color.lower()} color, this "
            f"{prod.lower()} is perfect for everyday use. Features include durable "
            f"construction, comfortable design, and modern aesthetics. Ideal for "
            f"{cat.lower()} enthusiasts looking for {adj.lower()} quality gear."
        )
        specs = (
            f"Material: {mat}. Color: {color}. Size: {size}. "
            f"Weight: {random.uniform(0.1, 5.0):.1f}kg. "
            f"Warranty: {random.randint(6, 36)} months. Origin: Imported."
        )

        docs.append({
            "product_id": sku,
            "name": name,
            "brand": brand,
            "category": cat,
            "sku": sku,
            "color": color,
            "size": size,
            "material": mat,
            "description": description,
            "specifications": specs,
            "price": price,
            "stock": stock,
            "attributes": {
                "rating": round(random.uniform(1.0, 5.0), 1),
                "reviews": random.randint(0, 500),
            },
        })
    return docs


def embed_documents(docs: list[dict], model_name: str, batch_size: int) -> list[dict]:
    """Add embeddings to documents using sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    print(f"Loading model: {model_name}...")
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()
    print(f"Model loaded. Embedding dimension: {dim}")

    # Build text to embed for each doc
    texts = []
    for doc in docs:
        text = f"{doc.get('name', '')} {doc.get('brand', '')} {doc.get('category', '')} {doc.get('description', '')}"
        texts.append(text.strip())

    total = len(texts)
    print(f"Embedding {total} documents in batches of {batch_size}...")

    all_embeddings = []
    start = time.time()

    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        embeddings = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        all_embeddings.extend(embeddings.tolist())

        done = min(i + batch_size, total)
        elapsed = time.time() - start
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        print(f"  {done}/{total} ({rate:.0f} docs/sec, ETA: {eta:.0f}s)", end="\r")

    print(f"\nEmbedding complete in {time.time() - start:.1f}s")

    for doc, emb in zip(docs, all_embeddings):
        doc["embedding"] = emb

    return docs


def main():
    parser = argparse.ArgumentParser(description="Pre-compute product embeddings")
    parser.add_argument("--input", type=str, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--generate", type=int, help="Generate N synthetic products instead of reading input")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="Sentence-transformer model name")
    parser.add_argument("--batch-size", type=int, default=256, help="Embedding batch size")
    args = parser.parse_args()

    if args.generate:
        print(f"Generating {args.generate} synthetic products...")
        docs = generate_synthetic(args.generate)
    elif args.input:
        print(f"Loading from {args.input}...")
        docs = []
        with open(args.input, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line))
        print(f"Loaded {len(docs)} documents")
    else:
        print("Error: specify --input or --generate", file=sys.stderr)
        sys.exit(1)

    docs = embed_documents(docs, args.model, args.batch_size)

    print(f"Writing to {args.output}...")
    with open(args.output, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"Done! {len(docs)} documents written to {args.output}")
    print(f"Embedding dimension: {len(docs[0]['embedding'])}")
    file_mb = sum(len(json.dumps(d)) for d in docs) / 1024 / 1024
    print(f"Approximate file size: {file_mb:.1f} MB")


if __name__ == "__main__":
    main()
