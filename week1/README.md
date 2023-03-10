 ## Applications of NLP
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

## NLP progress
> Speaker discussed about recent progress in NLP showing real world examples that is used in a daily basis.

1. **Google Translate**
    - a machine translation service developed by Google that can translate text, speech, images, and web pages between any combination of over 100 languages.
    - `Example:`  
    <img src='images/2.png' width=400>
2. **GPT-3**
    - GPT-3 (Generative pretrained Transformer 3) is a `SOTA` 3rd generation language model developed by OpenAI.
    - It is trained on a large corpus of text data (including books, articles, and web pages) and can generate high-quality natural text in response to prompt.
    - It has wide range of application such as text completion, summarization, translation, and question-answering with very few examples or no explicit training for a specific task.
    - `FINE TUNE on sample data:`

    ```
    **Train Step: FINE TUNE on custom train data (example shown below)**
    train_data1: I broke the window.
    label1: What did I break?  

    train_data2: I gracefully saved the day.
    label2: What did I gracefully save? and so on...
    
    **Test Step: Generate prediction for prompt (using trained model)**

    test_prompt1: I have John flowers.
    model_output1: Who did I give flowers do?

    test_prompt2: I gave her a rose and a guitar.
    model_output2: Who did I give a rose and a guitar to?

    ```
3. **ChatGPT**  
    - ChatGPT is a variant of GPT family of language models, developed by OpenAI which is trained on a large dataset of conversational data, and is able to generate responses that are appropriate to the context and tone of the conversation.
    - In contrast to GPT-3, ChatGPT is designed specifically for use in chatbots and conversational agents, whereas GPT-3 is more general purpose language model that can be used for a wide range of natural language processing tasks and it is not specifically designed for conversation.
    - `Chat GPT interface`: https://chat.openai.com/chat
    - <img src='images/1.png' width=500>


## Word Embeddings
> Before diving into NLP word, it is important to understand how we represent textual data which is not understood by the modern computers  . 

- Embeddings methods convert symbolic representation like words, emojis, and other features into meaningful numbers, capturing the underlying semantic relations between them.
- Word embeddings are numerical representation for words that capture their meaning, semantic relationships, and the different types of context they are used in, This enables computers to handle text data and perform various natural language processing tasks (such as text classification, text summarization, Question answering, Named Entity Recognition, etc).
- **Need of Word Embeddings**  
    - Dealing with textual data is problematic, since our computers, machine learning architectures and deep learning architectures cannot process strings or plain text in their raw form. They require numbers as inputs to perform any sort of job, either classification, regression, etc. Word embeddings are therefore necessary to best numerically represent textural input. 

- **Exmple**
    - Let us suppose we have task to compute similarity between emojis. Since computer does not understand emojis symbols directly, then how do we make computer understand about emojis? The answer is we construct emojis embedding, However there may be other approaches as well.
    - Suppose we choose four features `[spring, summer, autumn, winter]` to represent an emoji as a vector with 4 values i.e.
    - <img src='images/3.png' width='400'>
    - According to above matrix, we can embed each emoji according to the features they posses. It means, Tree emoji is represented by Spring, Summer, and Autumn, so on for others.
    - <img src='images/4.png' width='300'>
    - Now we can compute emojis similarity (say using Cosine) using the generated emojis vector shown above.

## One hot vectors 
> Assumption: _`Words are independent and unrelated entities`_
- One of the simplest ways to represent words is to consider them as independent and unrelated entities. 
- This can be achieved by creating a set of unique words from a corpus, for example: {..., chatgpt, ..., openai, ...}. 
- This approach provides a single representation for each word regardless of the context in which it is used.
-One way to represent these independent words is through a `One-hot vector` representation. 
    - This method assigns a unique binary vector to each word in a dictionary, where each element in the vector corresponds to a word in the dictionary.
    - The advantage of this representation is that it can be easily processed by a computer, as it transforms textual data into a numerical form.

However, the biggest `drawback` of this type of word representation is that it does not capture the similarity information between words i.e.
 - The dot product of two one-hot vectors representing two different words will always be zero, for example, let v and w be two one-hot vector representations of words, then their dot product will be 0, i.e., v.w = 0 = w.v.

In conclusion, while one-hot vector representation of words can be useful in processing textual data, it fails to capture the semantic relationships between words. This lack of similarity information is a major limitation of this type of word representation.

## Vectors from annotated discrete properties
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

## Distributional Semantics
>The key idea of distributional semantics is that words that occur in similar contexts tend to have similar meanings.   

- Example:  
    - Consider two words i.e. `tea` and `coffee`.
    - Let us suppose word `tea` is present in context such as _drank, the, pot, kettle, bad, delicious, oolong, hot, steam,.....,_.  
    - Since `tea` is similar to `coffee`, hence both of them will have similar distributions of context/surrounding words.
    - For such similar words, set of intersection of their context words will be high.  
- In distributional semantics, words are represented as vectors in a high dimensional space, where dimension corresponds to a feature or property of the words.  
      
## Word Vectors
- word vectors are distributed representations
- Also called as word embeddings or (neural word representations).
- Word vectors are basically a dense vector representation of words, where each components of vector represent some sort of discrete properties asscociated with words with semantic meaning.
- Common dimension of word vectors are: 50, 100, 200, 300  
- **Comparison with One hot Vector**  
    - As discussed above, One hot Encoding assumes words has no semantic meaning, where as word vectors carries semantic meaning.
    - One hot Vector is dependent on Vocab Size i.e. as Vocab Size increases the dimension of  One hot Vector also increases and leads to Sparse Vector, whereas Word Vectors are independent with Size of Vocabulary, and number of dimensions can be tuned as per requirements leading to dense vector.
- **Approaches to Word Vectors Computation**    

    1.`Frequency based Embedding`   
    - BOW, TF-IDF, Co-Occurence Vector  

    2.`Prediction based Embedding`  
    - Word2Vec(CBOW, Skip-Gram), 
    - Glove (Global Vectors for Word Representation)  
    - FastText
        - extension of skip-gram model. Handle out-of-vocabulary words effectively
    - ELMO (Embeddings from Language Models)  
    - BERT Embedding (Bidirectional Encoder Representation from Transformers)  

- **Example:**   
   - Let us consider we want word vector for words: `man`, `woman`, `king`, `queen`.
   - Assume we will use word vector with 4 dimension where,
        - 1st dimension = Gender
        - 2nd dimension = Royal
        - 3rd dimension = Age (adult?)
        - 4th dimension = Food
    
    -   |      Words/Features          | Gender | Royal | Age (adult?) | Food |
        | -------------- | ------ | ----- | ------------ | ---- |
        | Man      | -1     | 0.01  | 0.03         | 0.09 |
        | Woman    | 1      | 0.02  | 0.02         | 0.01 |
        | King     | -0.96  | 0.92  | 0.7          | 0.02 |
        | Queen    | 0.98   | 0.95  | 0.69         | 0.01 |
    - In above table, each rows represent words, and each column represents component/feature of word vector.
    - Using T-SNE algorithm we can visualize above word vectors by projecting 4D vectors to 3D or 2D.
        - You can use [Embedding Projector](https://projector.tensorflow.org/) for ease.


## word2vec
- Word2Vec (_Mikolov et al. 2013_) algorithm works on the idea of Distributional Semantics i.e. the word meaning can be understand by looking at the context it is present.  
- Word2Vec maxmimizes objective function by putting similar words nearby in space. 
- In the process of word vector computation, Word2Vec includes two different algorithms i.e. `CBOW` and `Skip-Gram`.
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
    
- **Optimization: Gradient Descent**   
    - `Recall:`
        - Gradient Descent is an algorithm to minimize cost function J(&theta;) by changing &theta;
        - From current value of &theta;, calculate gradient of J(&theta;), then take small step in the direction of negative gradient, and Repeat until optimial parameters &theta;.
        - <img src='images/5.png' width='400'>
        - This is the example when our objective function is convex, but in practice our objective function may not be convex.

    - Update RULE:
        - Update equation (in matrix notation)        
            - _&theta;(new)_ = _&theta;(old)_ - &alpha;&nabla;<sub>&theta;</sub>J(&theta;)
                - here, 
                    - &alpha; is learning rate or step size.
                    - &nabla;<sub>&theta;</sub>J(&theta;) is a gradient vectors obtained by taking gradients of cost function w.r.t parameters &theta;
        - Update equation (for single parameter)    
            - _&theta;<sub>j</sub><sup>new</sup>_ = _&theta;<sub>j</sub><sup>old</sup>_ - _&alpha; * ???/???x<sub>i</sub> J(&theta;)_

        - Pseudocode

            ```
            while True:
                theta_grad = evaluate_gradient(J, corpus, theta)  
                theta = theta - alpha * theta_grad
            ```
