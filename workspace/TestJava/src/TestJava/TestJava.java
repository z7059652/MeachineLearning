package TestJava;
import com.aliasi.matrix.SparseFloatVector;
import com.aliasi.matrix.Vector;
import com.aliasi.symbol.MapSymbolTable;
import com.aliasi.symbol.SymbolTable;
import com.aliasi.tokenizer.IndoEuropeanTokenizerFactory;
import com.aliasi.tokenizer.TokenizerFactory;
import com.aliasi.tokenizer.TokenFeatureExtractor;
import com.aliasi.util.Counter;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;

public class TestJava 
{
	public static Vector[] featureVectors(String[] texts,SymbolTable symbolTable)
	{
		Vector[] vectors = new Vector[texts.length];
		TokenizerFactory tokenizerFactory = new IndoEuropeanTokenizerFactory();
		TokenFeatureExtractor featureExtractor = new TokenFeatureExtractor(tokenizerFactory);
		for (int i = 0; i < texts.length; ++i) 
		{
			Map<String, Counter> featureMap = featureExtractor.features(texts[i]);
			vectors[i] = toVectorAddSymbols(featureMap, symbolTable,Integer.MAX_VALUE);
		}
		return vectors;
	}

	public static SparseFloatVector toVectorAddSymbols(Map featureVector, SymbolTable table,int numDimensions)
	{
		int size = (featureVector.size() * 3) / 2;
		Map vectorMap = new HashMap(size);
//		for (Map.Entry<String, Counter> entry : (Map.Entry<String, Counter>)featureVector.entrySet()) 
//		{
//			String feature = entry.getKey();
//			Number val = entry.getValue();
//			int id = table.getOrAddSymbol(feature);
//			vectorMap.put(new Integer(id), val);
//		}
		Iterator<Entry<String,Counter>> iter = featureVector.entrySet().iterator();
		while(iter.hasNext())
		{
			String feature = iter.next().getKey();
			Number val = iter.next().getValue();
			int id = table.getOrAddSymbol(feature);
			vectorMap.put(new Integer(id), val);
		}
		return new SparseFloatVector(vectorMap, numDimensions);
	}
	
	public static void main(String[] args) 
	{
		args = new String[]{"this is a book","a"};
		char mBoundaryChar = '\uFFFF';
//		SymbolTable symbolTable = new MapSymbolTable();
//		Vector[] vectors = featureVectors(args, symbolTable);
//		System.out.println("VECTORS");
//		for (int i = 0; i < vectors.length; ++i)
//			System.out.println(i + ") " + vectors[i]);
//		System.out.println(" SYMBOL TABLE");
		System.out.println(mBoundaryChar);
		}
	}

		// 寮�濮嬭缁�
//		for (int i = 0; i < CATEGORIES.length; i++)
//		{
//			File classDir = new File(TDIR, CATEGORIES[i]);
//			if (!classDir.isDirectory())
//			{
//				System.out.println("涓嶈兘鎵惧埌鐩綍=" + classDir);
//			}
//			// 璁粌鍣ㄩ亶鍘嗗垎绫绘枃浠跺す涓嬬殑鎵�鏈夋枃浠�
//			for (File file : classDir.listFiles())
//			{
//				String text = Files.readFromFile(file, "utf-8");
////				file.getPath();
////				String str = Files.readAllLines(file.getPath(), "utf-8");
////				Files.r
//				System.out.println("姝ｅ湪璁粌 " + CATEGORIES[i] + file.getName());
//				Classification classification = new Classification(CATEGORIES[i]);
//				Classified<CharSequence> classified = new Classified<CharSequence>(text, classification);
//				classifier.handle(classified);
//			} 
//		}	
		