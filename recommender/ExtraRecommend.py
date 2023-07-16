from elasticsearch import Elasticsearch


class ElasticsearchRecommender(object):
    def __init__(self, embeddings, sentences, host='localhost', port=9200, b=0.75, k1=1.2):
        self.es = Elasticsearch([{'host': host, 'port': port}])
        self.index_name = ""
        self.b = b
        self.k1 = k1

        # 创建索引
        self.es.indices.create(index=self.index_name)

        # 设置BM25算法
        self.es.indices.put_settings(index=self.index_name, body={
            'index': {
                'similarity': {
                    'my_similarity': {
                        'type': 'BM25',
                        'b': self.b,
                        'k1': self.k1
                    }
                }
            }
        })

        # 添加文档到索引中
        for i, embedding in enumerate(embeddings):
            doc = {
                'embedding': embedding.tolist(),
                'text': sentences[i]
            }
            self.es.index(index=self.index_name, id=i, body=doc)

    def recommend(self, new_embedding, topK):
        # 搜索相似的文档
        query = {
            'query': {
                'script_score': {
                    'query': {'match_all': {}},
                    'script': {
                        'source': 'cosineSimilarity(params.query_vector, "embedding") + 1.0',
                        'params': {'query_vector': new_embedding.tolist()}
                    }
                }
            },
            'size': topK,
            '_source': ['text']
        }
        res = self.es.search(index=self.index_name, body=query)

        # 解析搜索结果
        hits = res['hits']['hits']
        result_embeddings = [hit['_source']['embedding'] for hit in hits]
        result_texts = [hit['_source']['text'] for hit in hits]
        return result_embeddings, result_texts
