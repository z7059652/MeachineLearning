package TestJava;
import com.aliasi.classify.BernoulliClassifier;
import com.aliasi.classify.Classified;
import com.aliasi.classify.ConditionalClassifierEvaluator;
import com.aliasi.classify.Classification;
import com.aliasi.classify.ConditionalClassification;
import com.aliasi.classify.ConfusionMatrix;
import com.aliasi.classify.KnnClassifier;
import com.aliasi.classify.LogisticRegressionClassifier;
import com.aliasi.corpus.ObjectHandler;
import com.aliasi.corpus.XValidatingObjectCorpus;
import com.aliasi.io.Reporter;
import com.aliasi.io.Reporters;
import com.aliasi.matrix.KernelFunction;
import com.aliasi.matrix.PolynomialKernel;
import com.aliasi.stats.AnnealingSchedule;
import com.aliasi.stats.RegressionPrior;
import com.aliasi.tokenizer.RegExTokenizerFactory;
import com.aliasi.tokenizer.TokenizerFactory;
import com.aliasi.tokenizer.TokenFeatureExtractor;
import com.aliasi.util.AbstractExternalizable;
import com.aliasi.util.FeatureExtractor;
import com.aliasi.util.Files;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class PerceptronClassifier 
{
    static final String[] CATEGORIES = { "positive","negative"};

	private static String[] Path  = {"D:\\visual studio 2013\\Projects\\TextDeal\\Sentiment\\cn_sample_data\\sample.positive.temp.text.nonum.key.txt","D:\\visual studio 2013\\Projects\\TextDeal\\Sentiment\\cn_sample_data\\sample.negative.temp.text.nonum.key.txt"};
	private static String[] Split = {"}","Q"};

    public static void main(String[] args) throws Exception
    {
        PrintWriter progressWriter = new PrintWriter(System.out,true);
        progressWriter.println("Reading data.");
        int numFolds = 10;
        String outputAcceptCategory = new String();
        String outputRejectCategory = new String();
        String corpusAcceptCategory = "positive";
        XValidatingObjectCorpus<Classified<CharSequence>> corpus = new XValidatingObjectCorpus<Classified<CharSequence>>(numFolds);
        TokenizerFactory tokenizerFactory = new com.aliasi.tokenizer.NGramTokenizerFactory(2,4);
        FeatureExtractor<CharSequence> featureExtractor = new TokenFeatureExtractor(tokenizerFactory);
		BernoulliClassifier<CharSequence>  Berclassifier = new BernoulliClassifier<CharSequence>(featureExtractor);
		KnnClassifier<CharSequence> Knnclassifier = new KnnClassifier<CharSequence>(featureExtractor,5);
        KernelFunction kf = new PolynomialKernel(3);
        int numIterations = 10;
		for (int i = 0; i < CATEGORIES.length; i++)
		{   
			File file = new File(Path[i]);
			if(!file.exists() || file.isDirectory())
				throw new FileNotFoundException();
			BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(Path[i]),"UTF-8"));
			String temp=null;
	        StringBuffer sb=new StringBuffer();
	        temp=br.readLine();
	        while(temp!=null)
	        {
				temp=br.readLine();
				sb.append(temp);				
	        }
	        br.close();
			Classification classification = new Classification(CATEGORIES[i]);
	        String[] strarray = sb.toString().split(Split[i]);
	        for(int j = 0;j < strarray.length;j++)
	        {
				Classified<CharSequence> classified = new Classified<CharSequence>(strarray[j], classification);
				corpus.handle(classified);
				Knnclassifier.handle(classified);
	        }
		}
//        PerceptronClassifier<CharSequence> classifier = new PerceptronClassifier<CharSequence>(corpus,featureExtractor,kf,corpusAcceptCategory,numIterations,outcorpusAcceptCategory,outputRejectCategory);
		System.out.println("��ʼ���ɷ�����");
		String modelFile = "E:\\data\\category\\tclassifierKnn";
		ObjectOutputStream os = new ObjectOutputStream(new FileOutputStream(modelFile));
		Knnclassifier.compileTo(os);
    }

}
