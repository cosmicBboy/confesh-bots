from gensim.models.ldamodel import LdaModel
import pandas as pd

experiment_name = 'lda_model_specs_secrets_pdd9'
lda_model = LdaModel.load('./topic_modelling/models/%s.lda' % experiment_name)

topic_num = 1

data_dict = {}
col_names = []

for topics in lda_model.show_topics(num_topics=50, num_words=20):
    col_name = "Topic_%d" % topic_num
    col_names.append(col_name)
    pw = [pw.strip() for pw in topics.split('+')]
    data_dict[col_name] = [" - ".join(w.split('*')) for w in pw]
    topic_num += 1

data = pd.DataFrame(data_dict, columns=col_names)
data.to_csv("./topic_modelling/views/%s.csv" % experiment_name)
print data.head()
