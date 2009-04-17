package cs224n.langmodel;

import cs224n.langmodel.UnigramModel;
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
public class SimpleBackoffModel implements LanguageModel {
    
    private static final String START= "<S>";
    private static final String STOP = "</S>";
    
    private double[] coefs = {.78, 0.15, 0.07}; 

    private UnigramModel Unigram;
    private BigramModel Bigram;
    private TrigramModel Trigram;
    
    // -----------------------------------------------------------------------
    
    /**
     * Constructs a new, empty unigram language model.
     */
    public SimpleBackoffModel() {
	Unigram = new UnigramModel();
	Bigram = new BigramModel();
	Trigram = new TrigramModel();
    }
    
    /**
     * Constructs a unigram language model from a collection of sentences.  A
     * special stop token is appended to each sentence, and then the
     * frequencies of all words (including the stop token) over the whole
     * collection of sentences are compiled.
     */
    public SimpleBackoffModel(Collection<List<String>> sentences) {
	this();
	Unigram = new UnigramModel(sentences);
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
	/*double[] coefsTemp = {0.0,0.0,0.0};
	double bestP = 0.0;
	for(double i = 0.00; i <= 1.0; i+=0.1){
	    for(double j = 0.0; i <= 1.0; j+= 0.1){
		double k = 1.0 - i - j;
		if(k > 0){   
		    double Ptot = 0.0;
		    coefs[0] = i;
		    coefs[1] = j;
		    coefs[2] = k;
		    for(List<String> sentence : sentences){
			Ptot += getSentenceProbability(sentence);
		    }
		    if(Ptot > bestP){
			coefsTemp[0] = i;
			coefsTemp[1] = j;
			coefsTemp[2] = k;
			System.out.println("\nThe Coefs: " + coefs[0] +
					   " , " + coefs[1] + " , " +
					   coefs[2] + " = " + Ptot);
		    }
		}
	    }
	}
	coefs[0] = coefsTemp[0];
	coefs[1] = coefsTemp[1];
	coefs[2] = coefsTemp[2];
	System.out.println("\nThe Coefs: " + coefs[0] + " , " + coefs[1] +
	" , " + coefs[2]);*/
    }
    
    
    // -----------------------------------------------------------------------
    /**
     * Returns the probability, according to the model, of the word specified
     * by the argument sentence and index.  Smoothing is used, so that all
     * words get positive probability, even if they have not been seen
     * before.
     */
    public double getWordProbability(List<String> sentence, int index) {
	String wordN2 = sentence.get(index - 2).intern();
	String wordN1 = sentence.get(index - 1).intern();
	String word = sentence.get(index).intern();
	if(Trigram.getCount(wordN2, wordN1, word) > 0){
	    Trigram.getWordProbability(sentence, index);
	} else if(Bigram.getCount(wordN1, word) > 0) {
	    Bigram.getWordProbability(sentence, index);
	} else {
	    Unigram.getWordProbability(sentence,index);
	}
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
	return (coefs[2] * Unigram.checkModel() + 
		coefs[1] * Bigram.checkModel() +
		coefs[0] * Trigram.checkModel());
    }    
    /**
     * Returns a random word sampled according to the model.  A simple
     * "roulette-wheel" approach is used: first we generate a sample uniform
     * on [0, 1]; then we step through the vocabulary eating up probability
     * mass until we reach our sample.
     */
    public String generateWord(String prewordTwo, String prewordOne) {
	return Trigram.generateWord(prewordTwo, prewordOne);
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


