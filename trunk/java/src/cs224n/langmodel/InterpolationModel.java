package cs224n.langmodel;

import cs224n.langmodel.SmoothedUnigramLanguageModel;
import cs224n.langmodel.BigramModel;
import cs224n.langmodel.TrigramModel;

import cs224n.util.Counter;
import cs224n.util.CounterMap;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * A language model -- uses bigram counts
 */
public class InterpolationModel implements LanguageModel {
    
    private static final String START= "<S>";
    private static final String STOP = "</S>";
    

    private SmoothedUnigramLanguageModel Unigram;
    private BigramModel Bigram;
    private TrigramModel Trigram;

    private double[] Weights = {0.3, 0.4, 0.3};


    // -----------------------------------------------------------------------
    
    /**
     * Constructs a new, empty unigram language model.
     */
    public InterpolationModel() {
	Unigram = new SmoothedUnigramLanguageModel();
	Bigram = new BigramModel();
	Trigram = new TrigramModel();
    }
    
    /**
     * Constructs a unigram language model from a collection of sentences.  A
     * special stop token is appended to each sentence, and then the
     * frequencies of all words (including the stop token) over the whole
     * collection of sentences are compiled.
     */
    public InterpolationModel(Collection<List<String>> sentences) {
	this();
	Unigram = new SmoothedUnigramLanguageModel(sentences);
	Bigram = new BigramModel(sentences);
	Trigram = new TrigramModel(sentences); 
    }
	    
    // -----------------------------------------------------------------------
    
    /**
     * Constructs a unigram language model from a collection of sentences.  A
     * special stop token is appended to each sentence, and then the
     * frequencies of all words (including the stop token) over the whole
     * collection of sentences are compiled.
     */
    public void train(Collection<List<String>> sentences) {
	Unigram.train(sentences);
	Bigram.train(sentences);
	Trigram.train(sentences);
    }
    
    
    // -----------------------------------------------------------------------
    /**
     * Returns the probability, according to the model, of the word specified
     * by the argument sentence and index.  Smoothing is used, so that all
     * words get positive probability, even if they have not been seen
     * before.
     */
    public double getWordProbability(List<String> sentence, int index) {
	
	return (Weights[0] * Trigram.getWordProbability(sentence, index) +
		Weights[1] * Bigram.getWordProbability(sentence, index) +
		Weights[2] * Unigram.getWordProbability(sentence, index));
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
	return (Unigram.checkModel() + Bigram.checkModel() +
		Trigram.checkModel())/ 3.0;
    }    
    /**
     * Returns a random word sampled according to the model.  A simple
     * "roulette-wheel" approach is used: first we generate a sample uniform
     * on [0, 1]; then we step through the vocabulary eating up probability
     * mass until we reach our sample.
     */
    public String generateWord(String prewordTwo, String prewordOne) {
	double sample = Math.random();
	if(sample < Weights[0]){
	    return Trigram.generateWord(prewordTwo, prewordOne);
	} else if(sample < Weights[0] + Weights[1]){
	    return Bigram.generateWord(prewordOne);
	} else {
	    return Unigram.generateWord();
	}
	



	/*	double sum = 0.0;
	String result = Trigram.generateWord(Weights[0], sample);
	if(!reults.equals("*UNKNOWN*")) return result;
	sample -= Weights[0];
	result = Bigram.generateWord(Weights[1], sample);
	if(!reults.equals("*UNKNOWN*")) return result;
	return Unigram.generateWord(Weights[2], 
	*/
	/*for(String word : Trigram.wordCounter.keySet()) {
	    sum += Weights[0] * Trigram.wordCounter.getCount(word) / Trigram.getTotal();
	    if(sum > sample) {
		return word;
	    }
	}
	*/
	
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


