# jessica_nlp


```python
import re
import pickle
import jessica_nlp_re
import jessica_nlp_spark
import jessica_nlp_dl_train
from jessica_nlp_local_spark import sqlContext
```

pre-processing

```python
jessica_nlp_spark.text_json2text_normalized_json(
	input_json = "/Downloads/market_comments.json",
	output_json = "/Downloads/market_comments_norm.json",
	sqlContext = sqlContext)
```

match indicator

```python
re_negative = [
r'very poor',
r'poor',
r'lost',
r'loss',
r'underperformed',
r'bearish',
]

def negative_comment(text, \
	text_normalized, \
	entities = None):
	output = []
	for r,i in zip(re_negative, range(len(re_negative))):
		context_matched = jessica_nlp_re.text_normalized2text_entity_comb(\
		text_normalized, 
		indicator_re = r,\
		entities = entities)
		output += [{'entity':e, 'method':str(i)} for e in context_matched]
	return output

jessica_nlp_spark.text_normalized_entity_extraction(
	input_json = "/Downloads/market_comments_norm.json",
	output_json = 'market_comments_negative.json',
	entity_type = 'negative_comment',
	entity_extract_func = negative_comment,
	sub_entities = ['number', 'puntuation'],
	sqlContext = sqlContext)
```

train deep learning model

```python
jessica_nlp_spark.prepare_text_dl_input(\
	input_json = 'market_comments_negative.json',
	output_json = 'market_comments_negative_train.json',
	positive_indicator = 'negative_comment',\
	negative_sample_number = 10000,\
	sqlContext = sqlContext)

jessica_nlp_dl_train.train_text_categorization_model_from_json(\
	input_json = 'market_comments_negative_train.json',
	positive_weight_factor = 1,
	model_file = 'market_comments_negative.h5py',
	output_json_prediction = '/Downloads/market_comments_negative_prediction.json',
	output_json_recommend = '/Downloads/market_comments_negative_recommend.json',
	epochs = 4,
	sqlContext = sqlContext)
```

