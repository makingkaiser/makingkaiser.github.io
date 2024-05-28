# Statistical Evaluation

Statistical methods rely on comparing the presence of certain words or matching n-grams in the answer to the labeled ground truth. (N-gram: _**n**_ **continuous sequences of words**)

Because of the nature of LMs, people don’t typically use these methods anymore, unless for cases where word matching or accuracy of phrasing is _extremely important_, such as in legal settings or the medical field. Given that most people are using LMs for general-purpose understanding, these are not very helpful metrics because they only look to compare the actual word with ground truth data, and not any semantic meaning.

Exaggerated example: "Explain the concept of free will in the context of determinism.” an answer such as “determinism and free will inherently false paradigms of wrong thoughts” will still score highly using these methods because certain keywords such as “paradigm” or “free will” are present in the answer.

Nevertheless, I’ll include this section because I think it’s important to note how metrics have changed and developed over time.&#x20;

| Metric        | Use Case                                                 | Method                                                                                         | Why to Use                                                                                                 |
| ------------- | -------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| **ROUGE**     | Text Summarization                                       | Measures overlap of N-grams and Longest Common Subsequence (LCS) between summaries             | Popular for summarization tasks; captures content overlap between system-generated and reference summaries |
| **BLEU**      | Machine Translation                                      | Measures N-gram precision between candidate and reference translations                         | Most popular for translation tasks; captures word-by-word similarity                                       |
| **METEOR**    | Machine Translation, Text Generation                     | Calculates harmonic mean of unigram precision and recall, with a penalty for length mismatches | Can be used for various text generation tasks; balances precision and recall                               |
| **BERTScore** | Text Summarization, Machine Translation, Text Similarity | Computes cosine similarity between contextualized embeddings of words in sentences             | Captures semantic similarity between sentences; applicable to various NLP tasks                            |

This isn’t the whole list, but the others are in the same vein. I’ll talk about the more interesting evaluation methods, or benchmarks, on the next page.
