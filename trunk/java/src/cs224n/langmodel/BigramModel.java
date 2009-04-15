package cs224n.langmodel;

import cs224n.util.Counter;
import cs224n.util.CounterMap;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * A language model -- uses bigram counts
 */
public class BigramModel implements LanguageModel {
    
    private static final String START= "<S>";
    private static final String STOP = "</S>";
    
    private CounterMap<String, String> wordCounter;
    private double total;
    
    
    // -----------------------------------------------------------------------
    
    /**
     * Constructs a new, empty unigram language model.
     */
    public BigramModel() {
	wordCounter = new CounterMap<String, String>();
	total = Double.NaN;
    }
    
    /**
     * Constructs a unigram language model from a collection of sentences.  A
     * special stop token is appended to each sentence, and then the
     * frequencies of all words (including the stop token) over the whole
     * collection of sentences are compiled.
     */
    public BigramModel(Collection<List<String>> sentences) {
	this();
	train(sentences);
    }
    
    
    // -----------------------------------------------------------------------
    
    /**
     * Constructs a unigram language model from a collection of sentences.  A
     * special stop token is appended to each sentence, and then the
     * frequencies of all words (including the stop token) over the whole
     * collection of sentences are compiled.
     */
    public void train(Collection<List<String>> sentences) {
	wordCounter = new CounterMap<String, String>();
	for (List<String> sentence : sentences) {
	    List<String> stoppedSentence = new ArrayList<String>(sentence);
	    stoppedSentence.add(0, START);
	    stoppedSentence.add(STOP);
	    for(int i = 1; i < stoppedSentence.size(); i++){
		wordCounter.incrementCount(stoppedSentence.get(i-1),
					   stoppedSentence.get(i),
					   1.0);
	    }
	}
	total = wordCounter.totalCount();
    }
    
    
    // -----------------------------------------------------------------------
    public double getCount(String preword, String word) {
	Counter<String> counter = wordCounter.getCounter(preword);
        return counter.getCount(word);
    }


    private double getWordProbability(String preword, String word) {
	Counter<String> counter = wordCounter.getCounter(preword);
	double count = counter.getCount(word);
	double total = counter.totalCount();
	if (count == 0) {                   // unknown word
	    // System.out.println("UNKNOWN WORD: " + sentence.get(index));
	    return 1.0 / (total + 1.0);
	}
	return count / (total + 1);
    }
    
    /**
     * Returns the probability, according to the model, of the word specified
     * by the argument sentence and index.  Smoothing is used, so that all
     * words get positive probability, even if they have not been seen
     * before.
     */
    public double getWordProbability(List<String> sentence, int index) {
	String preword = sentence.get(index-1);
	String word = sentence.get(index);
	return getWordProbability(preword, word);
    }
    
    /**
     * Returns the probability, according to the model, of the specified
     * sentence.  This is the product of the probabilities of each word in
     * the sentence (including a final stop token).
     */
    public double getSentenceProbability(List<String> sentence) {
	List<String> stoppedSentence = new ArrayList<String>(sentence);
	stoppedSentence.add(0, START);
	stoppedSentence.add(STOP);
	double probability = 1.0;
	for (int index = 1; index < stoppedSentence.size(); index++) {
	    probability *= getWordProbability(stoppedSentence, index);
	}
	return probability;
    }
    
    /**
     * checks if the probability distribution properly sums up to 1
     */
    public double checkModel() {
	double sum = 0.0;
	int check = 10;
	int size = wordCounter.keySet().size();
	Object[] Words = (wordCounter.keySet()).toArray();

	for(int i = 0; i < check; i++){
	    int num = (int)(Math.random() * size);
	    String preword = (String)Words[num];
	    Counter<String> prewordCounter = wordCounter.getCounter(preword);
	    for(String word : prewordCounter.keySet()){	    
		sum += getWordProbability(preword, word);
	    }
	    sum += 1.0 / (prewordCounter.totalCount() + 1.0);
	}


	// since this is a bigram model, 
	// the event space is everything in the vocabulary (including START)
	// and a UNK token
	
	// this loop goes through the vocabulary (which includes START)
	//sum += wordCounter.totalCount();
	/*
	  for (String word : wordCounter.keySet()) {
	    sum += getWordProbability(word);
	}
	*/
	// remember to add the UNK. In this EmpiricalUnigramLanguageModel
	// we assume there is only one UNK, so we add...
	sum += 1.0 / (total + 1.0);
	
	return sum/check;
    }    
    /**
     * Returns a random word sampled according to the model.  A simple
     * "roulette-wheel" approach is used: first we generate a sample uniform
     * on [0, 1]; then we step through the vocabulary eating up probability
     * mass until we reach our sample.
     */
    public String generateWord(String preword) {
	double sample = Math.random();
	double sum = 0.0;
	Counter<String> subList = wordCounter.getCounter(preword);
	for (String word : subList.keySet()) {
	    sum += subList.getCount(word) / subList.totalCount();
	    if (sum > sample) {
		return word;
	    }
	}
	return "*UNKNOWN*";   // a little probability mass was reserved for unknowns
    }
    
    /**
     * Returns a random sentence sampled according to the model.  We generate
     * words until the stop token is generated, and return the concatenation.
     */
    public List<String> generateSentence() {
	List<String> sentence = new ArrayList<String>();
	String word = generateWord(START);
	while (!word.equals(STOP)) {
	    sentence.add(word);
	    word = generateWord(word);
	}
	return sentence;
    }
    
}


