<snippet>
  <content><![CDATA[
# ${1:Project Name}
The explosion in the availability of textual data prompts the need to devise a means to effectively compress textual data - Text Summarization. The project aims to deliver an effective text summarizer without comprimizing the intended semantics of the raw data. The proposed system employs clusting the Cosine-Similarity Matrix and sentence extraction from the each cluster.
## Proposed System
1. Data Preprocessing
    - Tokenizing 
    - Stopword Removal
    - Lemmatization 
2. Feature Extraction
    - Conversion of the input text to a TF.IDF matrix
3. Freature Tranformation
    - Transforming the TF.IDF matrix to a Cosine-Similarity Matrix
4. Clustering
    - K-Means
5. Sentence Extraction
]]></content>
  <tabTrigger>readme</tabTrigger>
</snippet>
