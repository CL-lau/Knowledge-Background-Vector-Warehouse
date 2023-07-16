class Recommender(object):
    def __init__(self, recommender):
        """
        :param recommender: 具体的推荐器。opt（AnnoyRecommender, FaissRecommender, HnswRecommender, BallTreeRecommender,
        ElasticsearchRecommender）
        交互类，封装各种搜索器。
        """
        self.recommender = recommender

    def recommend(self, new_embedding, topK):
        embeddings, sentences = self.recommender.recommend(new_embedding, topK)
        return embeddings, sentences

    def textRecommend(self, text, model, topK):
        # 这里的text只需要是str， 但是model参数为list
        new_embedding = model([text])
        embeddings, sentences = self.recommender.recommend(new_embedding, topK)
        return embeddings, sentences

