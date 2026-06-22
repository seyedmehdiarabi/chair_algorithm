from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


class ContextAwareReranker:

    def __init__(self):

        print("Loading Context Aware Reranker...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            "HooshvareLab/bert-base-parsbert-uncased"
        )

        self.model = AutoModel.from_pretrained(
            "HooshvareLab/bert-base-parsbert-uncased"
        )

        self.model.eval()

        print("Reranker READY ✔")


    def encode(self, text):

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )


        with torch.no_grad():

            outputs = self.model(**inputs)


        embedding = outputs.last_hidden_state[:,0,:]


        embedding = F.normalize(
            embedding,
            p=2,
            dim=1
        )

        return embedding



    def score(self, query, document):

        query_embedding = self.encode(query)

        doc_embedding = self.encode(document)


        similarity = F.cosine_similarity(
            query_embedding,
            doc_embedding
        )


        return float(similarity.item())



    def rerank(self, query, candidates):

        """
        candidates:
        [
          {
            question:"",
            answer:"",
            category:"",
            index:""
          }
        ]
        """


        reranked = []


        for item in candidates:


            document = (
                item.get("question","")
                + " "
                + item.get("answer","")
                + " "
                + item.get("category","")
            )


            score = self.score(
                query,
                document
            )


            item["rerank_score"] = score


            reranked.append(item)



        reranked.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )


        return reranked