import chromadb
from sentence_transformers import SentenceTransformer

class CaseRetriever:
    def __init__(self, db_path="backend/database/chroma_db"):
        # 1. Initialize persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)
        
        # 2. Use a open-source embedding model
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 3. Create or get vector collection
        self.collection = self.client.get_or_create_collection(
            name="radiology_cases"
        )
        
        # Seed initial dummy historical cases if collection is empty
        if self.collection.count() == 0:
            self._seed_sample_cases()

    def _seed_sample_cases(self):
        sample_cases = [
            {
                "id": "case_001",
                "text": "Patient present with persistent cough and fever. Chest radiograph shows opacity in the lower right lobe indicating localized Pneumonia. Treated with oral antibiotics.",
                "metadata": {"diagnosis": "Pneumonia", "severity": "Moderate"}
            },
            {
                "id": "case_002",
                "text": "Senior patient showing mild breathlessness. Radiograph reveals enlarged cardiac silhouette with bilateral basilar opacities indicative of Cardiomegaly and early Congestive Heart Failure.",
                "metadata": {"diagnosis": "Cardiomegaly", "severity": "Severe"}
            },
            {
                "id": "case_003",
                "text": "Follow-up scan after thoracic surgery showing pleural effusion on the left hemithorax. Blunting of costophrenic angle observed.",
                "metadata": {"diagnosis": "Effusion", "severity": "Mild"}
            }
        ]

        ids = [c["id"] for c in sample_cases]
        documents = [c["text"] for c in sample_cases]
        metadatas = [c["metadata"] for c in sample_cases]
        embeddings = self.embedder.encode(documents).tolist()

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def retrieve_similar_cases(self, query_text, top_k=2):
        query_embedding = self.embedder.encode([query_text]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        retrieved = []
        if results and "documents" in results:
            for i in range(len(results["documents"][0])):
                retrieved.append({
                    "case_id": results["ids"][0][i],
                    "summary": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i]
                })
        return retrieved


if __name__ == "__main__":
    retriever = CaseRetriever()
    test_results = retriever.retrieve_similar_cases("Patient with severe lung inflammation and high fever")
    print("Retrieved Cases:", test_results)