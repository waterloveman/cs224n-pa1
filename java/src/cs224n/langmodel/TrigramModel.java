package cs224n.langmodel;

import cs224n.util.Counter;
import cs224n.util.CounterMap;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Set;
import java.util.HashMap;

/**
 * A language model -- uses bigram counts
 */
public class TrigramModel implements LanguageModel {
    
    private static final String START= "<S>";
    private static final String STOP = "</S>";
    private static final String SEP = "=*=";
    
    private CounterMap<String, String> wordCounter;
    private Counter<String> totalMap;

    private BigramModel biModel;
    private HashMap<String, Double> preWordAlpha;
    private double discount = .75;
    
    private String SPLIT = "==";
    
    // -----------------------------------------------------------------------
    
    /**
     * Constructs a new, empty unigram language model.
     */
    public TrigramModel() {
	wordCounter = new CounterMap<String, String>();
	totalMap = new Counter<String>();
	biModel = new BigramModel();
	preWordAlpha = new HashMap<String, Double>();
    }
    
    /**
     * Constructs a unigram language model from a collection of sentences.  A
     * special stop token is appended to each sentence, and then the
     * frequencies of all words (including the stop token) over the whole
     * collection of sentences are compiled.
     */
    public TrigramModel(Collection<List<String>> sentences) {
	this();
	train(sentences);
    }
    
    public double getCount(String s1, String s2, String s3)
    {
	String key = concatStrings(s1, s2).intern();
	Counter<String> counter = wordCounter.getCounter(key);
	double count = counter.getCount(s3);
	return count;
    }
    
    public String[] getInitPhrase()
    {
	double pos = Math.random();
	Set<String> keys = wordCounter.keySet();
	String key = ((String)(keys.toArray())[(int)(keys.size() *
						     pos)]).intern();
	String[] ps = new String[2];
	ps[0] = key.substring(0, key.indexOf(SEP));
	ps[1] = key.substring(key.indexOf(SEP)+ SEP.length());
	return ps;
    }
    
    public Counter<String> getCounter(String s1, String s2){
	return wordCounter.getCounter(concatStrings(s1,s2));
    }


    public String concatStrings(String s1, String s2){
	return s1 + SEP + s2;
    }
    
    // -----------------------------------------------------------------------
    
    /**
     * Constructs a unigram language model from a collection of sentences.  A
     * special stop token is appended to each sentence, and then the
     * frequencies of all words (including the stop token) over the whole
     * collection of sentences are compiled.
     */
    public void train(Collection<List<String>> sentences) {
	biModel.train(sentences);

	wordCounter = new CounterMap<String, String>();
	for (List<String> sentence : sentences) {
	    List<String> stoppedSentence = new ArrayList<String>(sentence);
	    stoppedSentence.add(0, START);
	    stoppedSentence.add(0, START);
	    stoppedSentence.add(STOP);
	    for(int i = 2; i < stoppedSentence.size(); i++){
		String key = concatStrings(stoppedSentence.get(i-2),
					   stoppedSentence.get(i-1)).intern();
		if(stoppedSentence.get(i-1).length() == 0)
		    System.out.println(stoppedSentence.get(i-2));
		//System.out.println(key);
		wordCounter.incrementCount(key,
					   stoppedSentence.get(i).intern(),
					   1.0);
	    }
	}

	for(String word : wordCounter.keySet()){
	    totalMap.setCount(word, wordCounter.getCounter(word).totalCount());
	}
	
	for(String firstWord : wordCounter.keySet()){
	    //System.out.println(firstWord + " " + firstWord.split(SPLIT).length);
	    String[] words = firstWord.split(SPLIT);
	    String word2 = words.length == 1 ? "" : words[1]; 
	    String word1 = words[0];

	    Counter<String> thirdWords = wordCounter.getCounter(firstWord);
	    int firstTotal = (int)thirdWords.totalCount();
	    double sum = 0.0, denom = 1.0;
	    for (String thirdWord : thirdWords.keySet()){
		sum += (thirdWords.getCount(thirdWord) - discount) / firstTotal;
		denom -= biModel.getWordProbability(word2, thirdWord);
	    }
	    preWordAlpha.put(firstWord, (1-sum) / denom);
	    //System.out.println(firstWord + " " + sum + " " + denom + " " + (1-sum) / denom);
	}
	
    }
    
    
    // -----------------------------------------------------------------------
    
    private double getWordProbability(String preword, String word) {
	Counter<String> counter = wordCounter.getCounter(preword);
	double count = counter.getCount(word);
	double total = totalMap.getCount(preword);
	if (count == 0) {                   // unknown word
	    // System.out.println("UNKNOWN WORD: " + sentence.get(index));
	    //return 1.0 / (total + 1.0);
	    return (preWordAlpha.containsKey(preword) ? preWordAlpha.get(preword) : 1) * 
		biModel.getWordProbability(preword.split(SPLIT)[1], word);
	
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
	String word = sentence.get(index).intern();
	return getWordProbability(concatStrings(sentence.get(index-2).intern(),
						sentence.get(index-1).intern()).intern(),
				  word);
    }
    
    /**
     * Returns the probability, according to the model, of the specified
     * sentence.  This is the product of the probabilities of each word in
     * the sentence (including a final stop token).
     */
    public double getSentenceProbability(List<String> sentence) {
	List<String> stoppedSentence = new ArrayList<String>(sentence);
	stoppedSentence.add(0, START);
	stoppedSentence.add(0, START);
	stoppedSentence.add(STOP);
	double probability = 1.0;
	for (int index = 2; index < stoppedSentence.size(); index++) {
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
	    String preword = ((String)Words[num]).intern();
	    Counter<String> prewordCounter = wordCounter.getCounter(preword);
	    for(String word : biModel.uniModel.wordCounter.keySet()){	    
		sum += getWordProbability(preword, word);
	    }
	    sum += getWordProbability(preword, "*UNK*");
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
	//sum += 1.0 / (total + 1.0);
	return sum/check;
    }    
    /**
     * Returns a random word sampled according to the model.  A simple
     * "roulette-wheel" approach is used: first we generate a sample uniform
     * on [0, 1]; then we step through the vocabulary eating up probability
     * mass until we reach our sample.
     */
    public String generateWord(String prewordTwo, String prewordOne) {
	String preword = concatStrings(prewordTwo, prewordOne).intern();
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
	String oldWord = START;
	String word = generateWord(START,oldWord).intern();
	while (!word.equals(STOP)) {
	    sentence.add(word);
	    String temp  = generateWord(oldWord,word).intern();
	    oldWord = word;
	    word = temp;
	}
	return sentence;
    }
    
}
