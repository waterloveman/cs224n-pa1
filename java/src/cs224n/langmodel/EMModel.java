package cs224n.langmodel;

import cs224n.langmodel.UnigramModel;
import cs224n.langmodel.BigramModel;
import cs224n.langmodel.TrigramModel;

import cs224n.util.Counter;
import cs224n.util.CounterMap;
import java.util.ArrayList;
import java.util.Set;
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

    private double[] Lambdas = {0.7, 0.2, 0.1};

    private String wordPrint = null;

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
	int training = (int)(size * 0.8);
	int heldOut = size - training;
	Collection<List<String> > trainingSentences = new ArrayList<List<String>>();
	Collection<List<String> > heldOutSentences = new ArrayList<List<String>>();
	int i = 0;
	for(Iterator it=sentences.iterator(); it.hasNext(); ){
	    List<String> element = (List<String>)it.next();
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

	CounterMap<String, String> triC = Trigram.getCounterMap();
	
	double[] coefTots = {0.0, 0.0, 0.0};
	Set<String> keys = triC.keySet();
	System.out.println("");
	System.out.println("Total Trigrams: " + triC.keySet().size());
	int countThrough = 0;
	System.out.print("Progress");
	for(String key : keys)
	{
	    //System.out.println(key);
	    String[] wordParts = Trigram.split(key);
	    if(wordParts.length != 2) continue;
	    double[] tempC = getCoefs(wordParts[0], wordParts[1]);
	    if(tempC == null) continue;
	    countThrough++;
	    if(countThrough % 10000 == 0){
		System.out.print(".");
	    }
	    coefTots[0] += tempC[0];
	    coefTots[1] += tempC[1];
	    coefTots[2] += tempC[2];
	    break;
	}

	double tot = coefTots[0] + coefTots[1] + coefTots[2];

	Lambdas[0] = coefTots[0] / tot;
	Lambdas[1] = coefTots[1] / tot;
	Lambdas[2] = coefTots[2] / tot;
	
	//Scale
	
	System.out.println("\n=====LAMBDAS=====");
	System.out.println("Lambda 1: " + Lambdas[0]);
	System.out.println("Lambda 2: " + Lambdas[1]);
	System.out.println("Lambda 3: " + Lambdas[2]);
    }

    private double[] getCoefs(String wordN2, String wordN1)
    {
	double[] coefs = {0.7, 0.2, 0.1};
	Counter<String> trainCount = Trigram.getCounter(wordN2, wordN1);
	double totalCount = trainCount.totalCount();
	int runTimes = 10;
	Counter<String> heldoutCount = TrigramHeldout.getCounter(wordN2,
								 wordN1);
	/*
	  heldoutCount.incrementCount(Trigram.concatStrings(wordN2,
							  wordN1),
				    1.0);
	*/
	boolean print = false;
	if(heldoutCount.keySet().size() == 0){
	    runTimes = 1;
	    heldoutCount = Bigram.getCounter(wordN1);
	}
	
	for(int i = 0; i < runTimes; i++) {
	    //System.out.println("i: " + i);
	    //Trigram Expected
	    double TriExpected = 0.0;
	    double BiExpected = 0.0;
	    double UniExpected = 0.0;

	    double liklihood = 0.0;
	    for(String word : heldoutCount.keySet()){
		double TriP = coefs[0]*trainCount.getCount(word)/totalCount;
		double BiP =  coefs[1]*Bigram.getWordProbability(wordN1, word);
		double UniP = coefs[2]*Unigram.getWordProbability(word);

		liklihood += Math.pow(TriP + BiP + UniP,heldoutCount.getCount(word));

		
		double TriYP = TriP / (TriP + BiP + UniP);
		double BiYP = BiP / (TriP + BiP + UniP);
		double UniYP = UniP / (TriP + BiP + UniP);
		
		TriExpected += heldoutCount.getCount(word) * TriYP;
		BiExpected += heldoutCount.getCount(word) * BiYP;
		UniExpected += heldoutCount.getCount(word) * UniYP;
		if(!print) continue;
		if(wordPrint == null){
		    if(TriP > 0){
			wordPrint = word;
		    }
		} else if(wordPrint.equals(word)){ 
		    System.out.println(" =======WORD =========== ");
		    System.out.println("Word: " + word);
		    System.out.println("HO Count: " + heldoutCount.getCount(word));
		    System.out.println("Tri Prob: " + TriP);
		    System.out.println("Bi  Prob: " + BiP);
		    System.out.println("Uni Prob: " + UniP);
		    System.out.println("Prob From Tri: " + TriYP);
		    System.out.println("Prob From Bi : " + BiYP);
		    System.out.println("Prob From Uni: " + UniYP);
		    System.out.println("Exp From Tri: " + heldoutCount.getCount(word) * TriYP);
		    System.out.println("Exp From Bi : " + heldoutCount.getCount(word) * BiYP);
		    System.out.println("Exp From Uni: " + heldoutCount.getCount(word) * UniYP);
		}
	    }
	    double TotalExpected = TriExpected + BiExpected + UniExpected;
	    if(TotalExpected > 0){
		coefs[0] = TriExpected / TotalExpected;
		coefs[1] = BiExpected / TotalExpected;
		coefs[2] = UniExpected / TotalExpected;
	    }
	    if(print){
		System.out.println(" ======ITER " + i + "=====" );
		System.out.println("Liklihood :" + liklihood);
      		System.out.println("Lambda 1: " + coefs[0]);
		System.out.println("Lambda 2: " + coefs[1]);
		System.out.println("Lambda 3: " + coefs[2]);
	    }
	}
	return coefs;
    }



    
    
    // -----------------------------------------------------------------------
    /**
     * Returns the probability, according to the model, of the word specified
     * by the argument sentence and index.  Smoothing is used, so that all
     * words get positive probability, even if they have not been seen
     * before.
     */
    public double getWordProbability(List<String> sentence, int index) {
	return Lambdas[0] * Trigram.getWordProbability(sentence, index) +
	    Lambdas[1] * Bigram.getWordProbability(sentence, index) + 
	    Lambdas[2] * Unigram.getWordProbability(sentence, index);
    }

    public double getWordProbabilityTest(List<String> sentence, int index) {
	double ret = Lambdas[0] * Trigram.getWordProbability(sentence, index) +
	    Lambdas[1] * Bigram.getWordProbability(sentence, index) + 
	    Lambdas[2] * Unigram.getWordProbability(sentence, index);

	System.out.println(sentence.get(index) + " " + Unigram.getWordProbability(sentence, index) + " " + 
		   Bigram.getWordProbability(sentence, index) + " " + 
		   Trigram.getWordProbability(sentence, index) + " " + ret);
	return ret;
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

    public double getSentenceProbabilityTest(List<String> sentence) {
	List<String> stoppedSentence = new ArrayList<String>(sentence);
	stoppedSentence.add(0, START);
	stoppedSentence.add(0, START);
	stoppedSentence.add(STOP);
	double probability = 1.0;
	for (int index = 2; index < stoppedSentence.size(); index++) {
     	    probability *= getWordProbabilityTest(stoppedSentence, index);
	}
	return probability;
    }
    
    /**
     * checks if the probability distribution properly sums up to 1
     */
    public double checkModel() {
	//System.out.println(Unigram.checkModel() + " " + Bigram.checkModel() + " "  + Trigram.checkModel());
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


