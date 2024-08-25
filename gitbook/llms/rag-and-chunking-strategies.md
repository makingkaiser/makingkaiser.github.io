# 📑 RAG and chunking strategies

RAG (Retrieval Augmented Generation) originally was conceptualized as a paradigm that tried to mitigate the tendency of LLMs to hallucinate (making up false but plausible-sounding information). However, it has evolved into a standard that many people and companies have adopted for extending the capabilities with their own, customized, domain specific information. The basic idea is to augment the model with the relevant information required to answer the query at hand.

At its simplest, we can ‘append’ the relevant documents to the LLM together with the question itself in the prompt, and let it ‘search’ for itself the answer within the documents.

<figure><img src="https://lh7-rt.googleusercontent.com/docsz/AD_4nXd2KDv1hZpIBXdzCOYr9r-UlYfCJ4178orKc-Gdg-7PXs3Xty42rFMjleB1-v5hJV_V2s6SWvDjkbcq4LstJep7GsLs18lDL_V-ADxHEQ6oOOHiIV7yFOhJHfkcRcdh6fM3xiWuCJQo2I0b6TD4VjXnL3YF?key=Q-1Zs9sBWphZrTqHRnzY0w" alt=""><figcaption></figcaption></figure>

This works well with small documents that fit in within the LLM’s context window(the maximum length of text which an LLM can hold in its memory). But what about larger documents? In that case, we might run into the "needle in the haystack" problem. Because of the way a model's 'memory' works, the more you ask it to remember, the less likely it is to accurately recall an exact portion of a long paragraph or text.&#x20;

Working with long texts moves our problem closer towards the domain of traditional search and retrieval; I.e, we can’t entrust the entire task of search to the LLM.  (although it is tempting, as LLM capabilities get better on a monthly basis).

The most logical way to solve this issue is to split our long documents into smaller ones. This makes more sense if you think about the LLM being your average joe with access to an encyclopedia. If you asked it a question about cars, it would only need to look at sections that were related to cars, potentially significantly reducing the amount of text the model would need to process.

So we decide to split our documents into sections. What’s the best way to do so? As is in the real world, there is no straightforward answer. Other questions start to pop up: how small should my splits be? Do I split by chapter, or every sentence? What about on titles, for different subjects? All of these questions are largely dependent on the type of question and answer your RAG system is designed for.&#x20;

Splitting your text efficiently while ensuring good and relevant results requires a strategy unique to every use case and every type of documents. When I say documents, you can even transform other sources of information into text, such as diagrams or Excel sheets. However, to get the basic idea, for now let’s take a look at a few types of splitting.

**Character splitting** is the most basic form of splitting up your text. You simply divide your text into equal blocks of a certain size, which could be 50, 1000 characters etc. To mitigate the problems that come from splitting our documents this way, we do something called chunk overlap, which means that sequential chunks contains a certain number of overlapping characters. While this creates duplicate data across chunks, it preserves some context that would be lost otherwise.

We can also perform _document specific splitting_. By specifying certain separators depending on the type of document(Markdown, code, CSV, books with chapter headers), such as&#x20;

* <mark style="color:orange;">\n#{1,6} -</mark> Split by new lines followed by a header (H1 through H6)
* <mark style="color:orange;">\`\`\`</mark> - Code blocks
* <mark style="color:orange;">\n\\\\\*\\\\\*\\\\\*</mark>+ - Horizontal Lines
* and many others....&#x20;

We can make a split each time we encounter a common delimiter, one which according to our document type, is likely to mean a separation of a certain `topic.`Each of our chunks will be segmented in a way that more aligns to how it was intended to be in the original form.  With this method, we can split better according to what type of information is taken in.\


_Recursive character splitting_ adds one extra layer of complexity. It splits based on a series of separators. For eg, if we specify&#x20;

* <mark style="color:orange;">"\n\n"</mark> - Double new line, or most commonly paragraph breaks
* <mark style="color:orange;">"\n"</mark> - New lines
* <mark style="color:orange;">" "</mark> - Spaces
* <mark style="color:orange;">""</mark> - Characters

We split _first_ on double new lines. If the chunk is bigger than our desired size, i.e after a certain number of words we still haven't encountered a double new line, only _then do we_ split on new lines. If it is still bigger, it splits on spaces, and finally on characters. This accounts for the structure of our document by adding in more specificity on how we would like our documents to be split.

_**Semantic Chunking**_ is the most complicated chunking type. It aims to group chunks of words that contain similar meaning to each other. There are numerous ways of semantic chunking as people quantify ‘similar’ differently, and their methodologies are different as well. This starts to fall into the field of traditional NLP, and in recent times is not enough of a bottleneck for improving RAG systems compared to other issues. It also doesn’t help that this method introduces more latency compared to the simpler methods.&#x20;

### Semantic Search&#x20;

Now that we have chosen an appropriate method to _chunk_ our documents, what is the best way to _retrieve_ them?&#x20;

Naturally, the way we organize (chunk) our knowledge will directly affect the way we look for it (retrieval). However, let’s still look at simple methods of retrieving our chunks.\
\
The standard way is to encode each of our chunks into vector embeddings. To simplify, we turn our paragraphs or chunks of texts into a high dimensional ‘direction’, or vector. We can think of each of the dimensions as representing some sort of meaning. The purpose of doing so is that we can now start to compare chunks of text and determine how close two chunks are together in terms of **semantic similarity (semantic search)**. This loosely means, “_how closely are these 2 paragraphs related in terms of the same meaning?”_

<figure><img src="https://lh7-rt.googleusercontent.com/docsz/AD_4nXfVNPWdsPGS-WXLqeCujcH4qrLjkNRp5dw48S99OGBcInaHezDt7yvroIpHtYGWtxDdF1c9gA0K3ZdG3UaOxdRyJGS6jqI50Ep8woTNwBmHRvVlOh1LpZDiArcnO6BjCT_YIslvIcRfcvbl1TYQu946Q9Do?key=Q-1Zs9sBWphZrTqHRnzY0w" alt=""><figcaption><p>Similar vectors indicate closer meaning, while opposite vectors trend towards opposing meanings. </p></figcaption></figure>

Then, we compare our question (which is also a block of text!) to each of our document chunks, picking out the chunk(s) which closest match the meaning of our question. For each of our vectors, we use the cosine similarity between our query and chunk to check if it’s relevant to the question being asked. \


However, we can still go one step further(as always!), by shoring up some of the weaknesses of similarity search. The models that were used to turn our chunks into “meaning vectors” were trained on natural language. Sometimes, our documents contain pseudo-natural language, such as acronyms, which can be ambiguous. For example, _MVP_ can mean different things depending on who you ask. Think basketball fan vs startup CEO. This is where keyword search can come in to help.&#x20;

Also called “_full-text search_”, is built on old technology: BM25, powered by tf-idf (a way of representing text and weighing down words) is a baseline method from the 70’s that Google still uses when deciding what information to return to us. It’s especially strong on long documents containing domain-specific jargon, like medical texts, and has almost no computational overhead.&#x20;

### Reranking&#x20;

Now, we can use both semantic search(comparing vectors) and keyword search to ask our question. We then get 2 sets of differing results. One way to decide among both sets which ones are the most relevant in answering the query, is called the _reranker_.\
\
The idea is that after all the heavy lifting has been done by our more efficient models(cross-encoders) to handle our massive amount of text, we can now afford to use a computationally more expensive and powerful model (cross-encoder) to score the relevance of each chunk in the 2 curated sets to the query. This saves both time and cost for us.\
\
A short explanation of Bi vs Cross Encoders:&#x20;

Imagine you have 2 people looking at puzzle pieces(Bi-Encoder). One person looks at the first puzzle piece, and the other person looks at the second piece. Each person describes its piece without seeing the other one. Then, you compare their descriptions to guess if they might fit together. It's fast because each piece is analyzed independently, but it might miss some details about how well they actually fit. In our case, the 2 people are the encoders that look at our documents( sentence A/sentences) and another that looks at our query only(sentence B).&#x20;

In Cross-encoders, one person looks at both puzzle pieces together. This way, there is a more intuitive judgement to how the pieces interact and fit with each other. This method is more accurate in judging the match, but it's slower because they have to look at every possible pair of pieces one by one.\


<figure><img src="https://lh7-rt.googleusercontent.com/docsz/AD_4nXfG-UXZm98JFmI1pDGEMGgLpMWtW9BbnhTEgrXMeGT4PA3vxw_USSsA3mwSXZ2yyAE24njZxKqcRTv_u6ViQeNtU5VzkMYzd-7GrTw-k_vJeXI-3qYv6N6nDHmTZO14Y0wULXfJfmiUawntKFdqzWkh5PAn?key=Q-1Zs9sBWphZrTqHRnzY0w" alt=""><figcaption></figcaption></figure>

### Filtering&#x20;

One last step that we can do (for now) is to filter our documents based on their metadata. Typically, documents don’t exist on their own, and existing or inferred metadata can greatly improve our search results. Let’s take a look at this query:\
\
"Find the latest experimental results on CRISPR gene editing in human embryos from reputable journals in 2024."

This query challenges our semantic search and keyword search in a few ways. First, it needs to accurately capture and represent: "experimental results", "CRISPR gene editing", "reputable journals", and "2024" (Semantic search doesn’t do very well with numerical constraints.)&#x20;

Setting the number of retrieved documents too high could result in including irrelevant studies or outdated information, expecting the LLM to filter through them correctly.

Determining “reputable journals” is also a challenge - how would a general purpose embedding model know what is reputable?  This is a subjective criterion which is challenging for a simple embedding to capture.\
\
There are many other hurdles such as differentiating between experimental results, review papers, and theoretical studies, handling outdated citations and references that can be alleviated by filtering by metadata. Even when there isn’t any existing, we can use entity detection models such as GliNER, which can extract zero-shot entity types from text. We can then use the extracted entities to pre-filter our documents.&#x20;

Our final pipeline looks like this:

<figure><img src="https://lh7-rt.googleusercontent.com/docsz/AD_4nXfLh_mGG5miskfe_ZY-pZ8wLBSV70-BauvAjT4CpRA0Z5yMSSPey3Fmkf0DlyhtkZA77sEMCqM0OKcUQMDNNQTKRJdLQwZaUP10c-LyhR6ceWHOFGIbQwLeN59QGMEzLqIAH18wOfwrBvuvXZVnW1iBqmhC?key=Q-1Zs9sBWphZrTqHRnzY0w" alt=""><figcaption></figcaption></figure>

Which looks complicated, but is actually quite easy to implement (credits to [Ben Clavié](https://x.com/bclavie) ):

<figure><img src="https://lh7-rt.googleusercontent.com/docsz/AD_4nXda5dcHpNxs0S6VQY6Woc2ANESwzWcv1ygIZqC9vhwmx3jAnxHBnGDU10SB8Dt1voChpo3S-5jwcnJJS19kH1BDV8ld7MBar2sb4t9wGQcodGIHuxx9VCccsROjwPpDrHNbsEZhxo8ggT8P1lwaE7PUQLnB?key=Q-1Zs9sBWphZrTqHRnzY0w" alt=""><figcaption></figcaption></figure>

\
With all the basic components in place, this is actually quite a viable MVP for many projects. With further tuning and modification to a specific use case, we can reach performance levels on par with RAG models. (of course, how RAG models are benchmarked is a whole different story).&#x20;



I find it pretty cool that with all the capabilities that LLMs can demonstrate, albeit quite rarely and unpredictably, here we only use it as a tiny component in an information retrieval pipeline. In some ways, it feels like asking a PhD professor to do librarian duties. But with the unreliability and inherently unpredicatable nature of LLMs, this is the best we can do for now. Even as LLMs improve in terms of accuracy and reliability, I don’t think that RAG will ever disappear as a paradigm.&#x20;

After all, there is probably a good reason why PhD professors still have bookshelves.
