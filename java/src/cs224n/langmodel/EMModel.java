package cs224n.langmodel;

import cs224n.langmodel.UnigramModel;
import cs224n.langmodel.BigramModel;
import cs224n.langmodel.TrigramModel;

import cs224n.util.Counter;
import cs224n.util.CounterMap;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Iterator;
/**
 * A language model -- uses bigram counts
 */
public class EMModel implements LanguageModel {
    
    private static final String START= "<S>";
    private static final String STOP = "</S>";

    private UnigramModel Unigram;
    private BigramModel Bigram;
    private TrigramModel Trigram;
    
    private UnigramModel UnigramHeldout;
    private BigramModel BigramHeldout;
    private TrigramModel TrigramHeldout;

    // -----------------------------------------------------------------------
    
    /**
     * Constructs a new, empty unigram language model.
     */
    public EMModel() {
	Unigram = new UnigramModel();
	Bigram = new BigramModel();
	Trigram = new TrigramModel();

	UnigramHeldout = new UnigramModel();
	BigramHeldout = new BigramModel();
	TrigramHeldout = new TrigramModel();
	


    }
    
    /**
     * Constructs a unigram language model from a collection of sentences.  A
     * special stop token is appended to each sentence, and then the
     * frequencies of all words (including the stop token) over the whole
     * collection of sentences are compiled.
     */
    public EMModel(Collection<List<String>> sentences) {
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
		
	int size = sentences.size();
	int training = (int)(size * 0.9);
	int heldOut = size - training;
	Collection<List<String> > trainingSentences = new ArrayList<List<String>>();
	Collection<List<String> > heldOutSentences = new ArrayList<List<String>>();
	int i = 0;
	for(Iterator it=sentences.iterator(); it.hasNext(); ){
	    List<String>  element = (List<String>)it.next();
	    if(i <= training){
		trainingSentences.add(element);
	    } else {
		heldOutSentences.add(element);
	    }
	    i++;
	}

	Unigram.train(trainingSentences);
	Bigram.train(trainingSentences);
	Trigram.train(trainingSentences);
	
	UnigramHeldout.train(heldOutSentences);
	BigramHeldout.train(heldOutSentences);
	TrigramHeldout.train(heldOutSentences);
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
	String wordN = sentence.get(index).intern();
	
	double[] coefs = {0.7, 0.2, 0.1};
	Counter<String> trainCount = Trigram.getCounter(wordN2, wordN1);
	double totalCount = trainCount.totalCount();
	Counter<String> heldoutCount = TrigramHeldout.getCounter(wordN2,
								 wordN1);
	heldoutCount.incrementCount(Trigram.concatStrings(wordN2,
							  wordN1),
				    1.0);
	for(int i = 0; i < 1; i++) {
	    //Trigram Expected
	    double TriExpected = 0.0;
	    double BiExpected = 0.0;
	    double UniExpected = 0.0;
	    for(String word : heldoutCount.keySet()){
		double TriP = coefs[0]*trainCount.getCount(word)/totalCount;
		double BiP =  coefs[1]*Bigram.getWordProbability(wordN1, word);
		double UniP = coefs[2]*Unigram.getWordProbability(word);
		
		double TriYP = TriP / (TriP + BiP + UniP);
		double BiYP = BiP / (TriP + BiP + UniP);
		double UniYP = UniP / (TriP + BiP + UniP);
		
		TriExpected += heldoutCount.getCount(word) * TriYP;
		BiExpected += heldoutCount.getCount(word) * BiYP;
		UniExpected += heldoutCount.getCount(word) * UniYP;
	    }
	    double TotalExpected = TriExpected + BiExpected + UniExpected;
	    /*
	      System.out.println("Actual: " + heldoutCount.totalCount() +
			       "   Got: " + TotalExpected);
	    /*
	      if(TotalExpected != heldoutCount.totalCount()){
		System.out.println("InValid Reestimation");
	    }
	    */
	    if(TotalExpected > 0){
		coefs[0] = TriExpected / TotalExpected;
		coefs[1] = BiExpected / TotalExpected;
		coefs[2] = UniExpected / TotalExpected;
	    }
	}
	
	double P1 = coefs[0] * Trigram.getWordProbability(sentence, index);
	double P2 = coefs[1] * Bigram.getWordProbability(sentence, index);
	double P3 = coefs[2] * Unigram.getWordProbability(sentence,
	index);
	//System.out.println("PROBS: " + P1 + "," + P2 + "," + P3);
	return P1 + P2 + P3;
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
	    double p = getWordProbability(stoppedSentence, index);
	    if(!(p > 0)) {
		//System.out.println("PROB: " + p);
	    }
	    probability *= p;
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


