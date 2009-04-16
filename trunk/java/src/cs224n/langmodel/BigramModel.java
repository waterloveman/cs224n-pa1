package cs224n.langmodel;

import cs224n.util.Counter;
import cs224n.util.CounterMap;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.HashMap;

/**
 * A language model -- uses bigram counts
 */
public class BigramModel implements LanguageModel {
    
    private static final String START= "<S>";
    private static final String STOP = "</S>";
    
    private CounterMap<String, String> wordCounter;
    private Counter<String> totalMap;
    public UnigramModel uniModel;
    private HashMap<String, Double> preWordAlpha;
    private double discount = .75;
    // -----------------------------------------------------------------------
    
    /**
     * Constructs a new, empty unigram language model.
     */
    public BigramModel() {
	uniModel = new UnigramModel();
	wordCounter = new CounterMap<String, String>();
	totalMap = new Counter<String>();
	preWordAlpha = new HashMap<String, Double>();
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
	uniModel.train(sentences);

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
	for(String word : wordCounter.keySet()){
	    totalMap.setCount(word, wordCounter.getCounter(word).totalCount());
	}
	
	for(String firstWord : wordCounter.keySet()){
	    Counter<String> secondWords = wordCounter.getCounter(firstWord);
	    int firstTotal = (int)secondWords.totalCount();	
	    double sum = 0.0;
	    double denom = 1.0;

	    for(String secondWord: secondWords.keySet()){
		sum += (secondWords.getCount(secondWord) - discount) / firstTotal;
		denom -= uniModel.getWordProbability(secondWord);
	    }
	   
	    preWordAlpha.put(firstWord, (1 - sum) / denom);
	}
    }
    
    
    // -----------------------------------------------------------------------
  
    public Counter<String> getCounter(String preword){
	return wordCounter.getCounter(preword);
    }

    public double getCount(String preword, String word) {
	Counter<String> counter = wordCounter.getCounter(preword);
        return counter.getCount(word);
    }


    public double getWordProbability(String preword, String word) {
	Counter<String> counter = wordCounter.getCounter(preword);
	double count = counter.getCount(word);
	double total = totalMap.getCount(preword);
	if (count == 0) {                   // unknown word
	    return (preWordAlpha.containsKey(preword) ? preWordAlpha.get(preword) : 1) * uniModel.getWordProbability(word);// System.out.println("UNKNOWN WORD: " + sentence.get(index));
	}
	return (count - discount) / total;
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
	    for(String word : uniModel.wordCounter.keySet())
		sum += getWordProbability(preword, word);
	    sum += getWordProbability(preword, "*UNK*");
	}	
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


