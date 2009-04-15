package cs224n.langmodel;

import cs224n.util.Counter;
import cs224n.util.Pair;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * A dummy language model -- uses empirical unigram counts, plus a single
 * ficticious count for unknown words.  (That is, we pretend that there is
 * a single unknown word, and that we saw it just once during training.)
 *
 * @author Dan Klein
 */
public class SmoothedUnigramLanguageModel implements LanguageModel {
    
    private static final String STOP = "</S>";
    
    public Counter<String> wordCounter;
    private Pair<Double, Double> regFunc;
    private int[] frequencyCount;
    private double total;
    
    
    // -----------------------------------------------------------------------
    
    /**
     * Constructs a new, empty unigram language model.
     */
    public SmoothedUnigramLanguageModel() {
	wordCounter = new Counter<String>();
	total = Double.NaN;
    }
    
    /**
     * Constructs a unigram language model from a collection of sentences.  A
     * special stop token is appended to each sentence, and then the
     * frequencies of all words (including the stop token) over the whole
     * collection of sentences are compiled.
     */
    public SmoothedUnigramLanguageModel(Collection<List<String>> sentences) {
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
	wordCounter = new Counter<String>();
	double maxCount = 0;

	for (List<String> sentence : sentences) {
	    List<String> stoppedSentence = new ArrayList<String>(sentence);
	    stoppedSentence.add(STOP);
	    for (String word : stoppedSentence) {
		wordCounter.incrementCount(word, 1.0);
		if (wordCounter.getCount(word) > maxCount)
		    maxCount = wordCounter.getCount(word);
	    }
	}
	frequencyCount = new int[(int)maxCount+1];

	for(String key : wordCounter.keySet())
	    frequencyCount[(int)wordCounter.getCount(key)]++;

	total = wordCounter.totalCount();

	ArrayList<Pair<Double, Double> > dataPoints = new ArrayList<Pair<Double, Double> >();

	for(int i = 1; i < frequencyCount.length ;i++){	    
	    if(frequencyCount[i] == 0)
		continue;
	    double firstVal = Math.log(i), secondVal = Math.log(frequencyCount[i]);
	    dataPoints.add(new Pair<Double, Double>(firstVal, secondVal));
	}

	linearRegression(dataPoints);

	total = 0;
	for(String key : wordCounter.keySet()){
	    int count = (int)wordCounter.getCount(key);
	    total += (count + 1) * simpleCount(count + 1) / simpleCount(count);
	}
	total += simpleCount(1);
    }

    private void linearRegression(ArrayList<Pair<Double, Double> > dataPoints){
	double a = 1, b = 1;
	double alpha = .0001;
	while(true){
	    double asum=0, bsum=0;
	    for(Pair<Double, Double> p : dataPoints){
		double guess = p.getSecond() - (a + b * p.getFirst());

		asum += guess;
		bsum += guess * p.getFirst();
	    }
	    if(Math.abs(asum) < .001 && Math.abs(bsum) < .001)
		break;
	    a += alpha * asum;
	    b += alpha * bsum;
	}
	regFunc = new Pair<Double,Double>(Math.pow(Math.E, a), b);
    }

    private double simpleCount(int freq){
	double ret = regFunc.getFirst() * Math.pow(freq, regFunc.getSecond());
	return ret;
    }
    
    // -----------------------------------------------------------------------
    
    private double getWordProbability(String word) {
	int count = (int)wordCounter.getCount(word);
	if (count == 0) {                   // unknown word
	    return simpleCount(1) / total;
	}
	return (count + 1) * simpleCount(count+1) / simpleCount(count) / total;
    }
    
    /**
     * Returns the probability, according to the model, of the word specified
     * by the argument sentence and index.  Smoothing is used, so that all
     * words get positive probability, even if they have not been seen
     * before.
     */
    public double getWordProbability(List<String> sentence, int index) {
	String word = sentence.get(index);
	double ret = getWordProbability(word);
	return ret;
    }
    
    /**
     * Returns the probability, according to the model, of the specified
     * sentence.  This is the product of the probabilities of each word in
     * the sentence (including a final stop token).
     */
    public double getSentenceProbability(List<String> sentence) {
	List<String> stoppedSentence = new ArrayList<String>(sentence);
	stoppedSentence.add(STOP);
	double probability = 1.0;
	for (int index = 0; index < stoppedSentence.size(); index++) {
	    probability *= getWordProbability(stoppedSentence, index);
	}
	return probability;
    }
    
    /**
     * checks if the probability distribution properly sums up to 1
     */
    public double checkModel() {
	double sum = 0.0;
	// since this is a unigram model, 
	// the event space is everything in the vocabulary (including STOP)
	// and a UNK token
	
	// this loop goes through the vocabulary (which includes STOP)
	for (String word : wordCounter.keySet()) {
	    sum += getWordProbability(word);
	}
	// remember to add the UNK. In this EmpiricalUnigramLanguageModel
	// we assume there is only one UNK, so we add...
	sum += simpleCount(1) / total;
	return sum;
    }
    
    public double getTotal() {
	return total;
    }
    
    /**
     * Returns a random word sampled according to the model.  A simple
     * "roulette-wheel" approach is used: first we generate a sample uniform
     * on [0, 1]; then we step through the vocabulary eating up probability
     * mass until we reach our sample.
     */
    public String generateWord() {
	double sample = Math.random();
	double sum = 0.0;
	for (String word : wordCounter.keySet()) {
	    sum += (wordCounter.getCount(word) / total);
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
	String word = generateWord();
	while (!word.equals(STOP)) {
	    sentence.add(word);
	    word = generateWord();
	}
	return sentence;
    }
    
}


