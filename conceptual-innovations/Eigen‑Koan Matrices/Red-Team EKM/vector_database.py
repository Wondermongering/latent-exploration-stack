def store_findings_in_vectordb(findings, embedding_model):
    """Store vulnerability findings in vector database for clustering"""
    import chromadb
    
    # Initialize ChromaDB client
    client = chromadb.Client()
    collection = client.create_collection("vulnerability_findings")
    
    # Add documents with embeddings
    for i, finding in enumerate(findings):
        # Create embedding of the vulnerability
        trigger_prompt = finding["trigger_prompt"]
        embedding = embedding_model.embed(trigger_prompt)
        
        # Store in vector DB with metadata
        collection.add(
            ids=[f"vuln_{i}"],
            embeddings=[embedding],
            documents=[trigger_prompt],
            metadatas=[{
                "safety_score": finding["safety_score"],
                "path": str(finding["path"]),
                "timestamp": datetime.datetime.now().isoformat()
            }]
        )
    
    return collection
