**Applications of NLP**
- Machine Translation
    - translates from one language to another.
    - It is Ubiquitous
    - Difficulties:
        - Translating long text
- Question answering and information retrieval
- Summarization and analysis of text
- Speech to text
    - Automatic transcriptions of spoken or signed language (audio, or video) to textual representations.
    - It's not in priorities in this course (i.e. in CS224n)

**One hot vectors**  
- One of the simplest ways to represent words is to consider them as independent and unrelated entities. 
- This can be achieved by creating a set of unique words from a corpus, for example: {..., chatgpt, ..., openai, ...}. 
- This approach provides a single representation for each word regardless of the context in which it is used.
-One way to represent these independent words is through a `One-hot vector` representation. 
    - This method assigns a unique binary vector to each word in a dictionary, where each element in the vector corresponds to a word in the dictionary.
    - The advantage of this representation is that it can be easily processed by a computer, as it transforms textual data into a numerical form.

However, the biggest drawback of this type of word representation is that it does not capture the similarity information between words i.e.
 - The dot product of two one-hot vectors representing two different words will always be zero, for example, let v and w be two one-hot vector representations of words, then their dot product will be 0, i.e., v.w = 0 = w.v.

In conclusion, while one-hot vector representation of words can be useful in processing textual data, it fails to capture the semantic relationships between words. This lack of similarity information is a major limitation of this type of word representation.

**Vectors from annotated discrete properties**
- It refers to the representation of entities or objects (words in our scenario) as numerical vectors based on their annotated discrete properties or features.
- The objective is to capture the important information about each entity in a numerical format, so that mathematical operations () can be performed on them.
    - Example:
        - Let's suppose we want to generate vector for the word `tea` and assume only `plural noun`, `3rd singular verb`, `hyponynm-of-beverage`, and `synonym-of-chai` as discrete properties for the sake of simiplicity.
        - Vector representation of word `tea` is:  **[0, 0, 1, 1]**
            - Here,
                - 1st vector component = plural noun  
                - 2nd vector component = 3rd singular verb  
                - 3rd vector component = hyponym-of-beverage
                - 4th vector component = synonym-of-chai

- To annotate the information about words, resources such as `WordNet` or `UniMorph` can be used. These resources provide information about various linguistic properties of words, which can be used to generate the vector representations.   
- **Drawbacks:**
    - Human-annotated resources (like WordNet) are always lacking in vocabulary and updating these resources may be costly.  
    - Tradeoff between dimensionality and utility of vectors, 
        - It refers to the size of the vector representation and its usefulness i.e. dimensions can quickly become very large, much larger than size of the vocabulary, inorder to capture all the different categories and properties of the words.
        - Also, such high dimensions sparse vector representations doesn't work well with modern neural networks that tend to operate on dense vectors.
    - Human ideas of optimal text representations tends to underperform when dealing with large amounts of data.

**Distributional Semantics**  
The key idea of distributional semantics is that words that occur in similar contexts tend to have similar meanings.  
Example:  
    - Consider two words i.e. `tea` and `coffee`.
    - Let us suppose word `tea` is present in context such as _drank, the, pot, kettle, bad, delicious, oolong, hot, steam,.....,_.  
    - Since `tea` is similar to `coffee`, hence both of them will have similar distributions of context/surrounding words.  
- In distributional semantics, words are represented as vectors in a high dimensional space, where dimension corresponds to a feature or property of the words.  
- The vectors representation of words are computed using two approach i.e.  
    1.`Frequency based Embedding`   
    - BOW, TF-IDF, Co-Occurence Vector  

    2.`Prediction based Embedding`  
    - Word2Vec(CBOW, Skip-Gram), 
    - Glove (Global Vectors for Word Representation)  
    - FastText
        - extension of skip-gram model. Handle out-of-vocabulary words effectively
    - ELMO (Embeddings from Language Models)  
    - BERT Embedding (Bidirectional Encoder Representation from Transformers)  
      
**Word Vectors**  
- word vectors are distributed representations
- Also called as word embeddings or (neural word representations).
- It's a basic problems in NLP.

**word2vec**
- Word2Vec algorithm works on the idea of Distributional Semantics i.e. the word meaning can be understand by looking at the context it is present.
- **Architecture (Skip-Gram)**  
    - `Input Layer`    
        - The input layer takes the one-hot representation of the center word and passes it to a hidden layer.
    - `Hidden Layer`  
        - The hidden layer receives the input from the input layer and performs matrix multiplication with the weight matrix U.
        - Then output is passed through the activation function to obtain the embedding representation of the center word.
    - `Output Layer`  
        - The output layer receives the embedding representation of the center word and performs a matrix multiplication with the weight matrix V. The result of this multiplication is then passed through a softmax activation to obtain the predicted context words.

- **objective functions**
    - Likelihood in converted to Log likelihood because, 
        - The probability of the data given the parameters (P(X|theta)) is referred to as the likelihood.   
        - rather than working in products, it's easier to work with sums.
        - we want to minimize the likelhood functions, rather than maximizing it so minus sign is present in the objective functions.

- **How to calculate P(wt+j | wt; theta)**
    - we will use two word vectors per word w i.e. we're actually going to give two word vectors for each words.
        - one word vector when it's used as the center word, and a different word vector that's used as a context word.
        - This is done because it's just simplifies the math and the optimization, and makes building word vectors a lot easier

- softmax is a way to convert number to probabilites
    - we use exp because to convert any negative numbers to positive since probability can't be negative.
    - 