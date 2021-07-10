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
sqlContext.read.json("market_comments.json").registerTempTable("market_comments")
sqlContext.sql(u"""
	SELECT *, MD5(text) AS document_id
	FROM market_comments
	""").write.mode("Overwrite").json("/Downloads/market_comments.json")
  ```
