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
