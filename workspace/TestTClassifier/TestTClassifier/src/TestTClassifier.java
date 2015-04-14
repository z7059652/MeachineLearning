import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.InputStreamReader;
import java.text.NumberFormat;

import com.aliasi.classify.*;

public class TestTClassifier
{
	private static String[] CATEGORIES = { "negative","positive"};
	private static String[] Path  = {"D:\\visual studio 2013\\Projects\\TextDeal\\Sentiment\\test.label.cn.negative.key.txt","D:\\visual studio 2013\\Projects\\TextDeal\\Sentiment\\test.label.cn.positive.key.txt"};
	private static String[] Split = {">",">"};
	public static void main(String[] args) throws ClassNotFoundException, IOException
	{
		// TODO 自动生成的方法存根
		//分类器模型存放地址  

        String modelFile = "E:\\data\\category\\tclassifierNGB"; 
        ScoredClassifier<String> compiledClassifier = null;
        try 
        {  
            ObjectInputStream oi = new ObjectInputStream(new FileInputStream(modelFile));  
            compiledClassifier = (ScoredClassifier<String>) oi.readObject();  
            oi.close();  
        } 
        catch (IOException ie)
        {  
            System.out.println("IO Error: Model file " + modelFile + " missing");  
        }  
        
     // 遍历分类目录中的文件测试分类准确度  
        ConfusionMatrix confMatrix = new ConfusionMatrix(CATEGORIES);  
        NumberFormat nf = NumberFormat.getInstance();  
        nf.setMaximumIntegerDigits(1);  
        nf.setMaximumFractionDigits(3);  
        int[] ncount = new int[2];
        int[] uncorrect = new int[2];
        for(int j = 0;j <= 1;j++)
        {
			File file = new File(Path[j]);
			if(!file.exists() || file.isDirectory())
				throw new FileNotFoundException();
			BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(Path[j]),"UTF-8"));
			String temp=null;
	        StringBuffer sb=new StringBuffer();
	        temp=br.readLine();
	        while(temp!=null)
	        {
				temp=br.readLine();
				sb.append(temp);				
	        }
	        br.close();
	        String[] strarray = sb.toString().split(">");
	        System.out.println(CATEGORIES[j] + "：总长度为：" +strarray.length);
//	        String text = "根本不是MTV，都是演唱会版本，VCD画质，不值，千万别买！";
//	        int ncount = 0;
	        for (int i = 0;i<strarray.length;i++)
	        {  
	            ScoredClassification classification = compiledClassifier.classify(strarray[i]);  //.subSequence(0, strarray[i].length())
	            confMatrix.increment(CATEGORIES[j], classification.bestCategory());  
//	            System.out.println("最适合的分类: " +strarray[i]+": " +classification.bestCategory());  
	            if(classification.bestCategory().contains(CATEGORIES[j]))
	            {
	            	ncount[j]++;
	            }
	            if(classification.bestCategory().contains(CATEGORIES[1-j]))
	            {
	            	uncorrect[j]++;
	            }
	        } 
	        System.out.println(CATEGORIES[j] + ":正确总数：" + ncount[j]); 
	        System.out.println(CATEGORIES[j] + ":错误总数：" + uncorrect[j]); 
        }
        
//            System.out.println("总数：" + j); 
//        }   
//        System.out.println("--------------------------------------------");  
//        System.out.println("- 结果 ");  
//        System.out.println("--------------------------------------------");  
//        int[][] imatrix = confMatrix.matrix();  
//        StringBuffer sb = new StringBuffer();  
//        sb.append(StringTools.fillin("CATEGORY", 10, true, ' '));  
//        for (int i = 0; i < CATEGORIES.length; i++)  
//            sb.append(StringTools.fillin(CATEGORIES[i], 8, false, ' '));  
//        System.out.println(sb.toString());  
//        for (int i = 0; i < imatrix.length; i++)
//        {  
//            sb = new StringBuffer();  
//            sb.append(StringTools.fillin(CATEGORIES[i], 10, true, ' ',10 - CATEGORIES[i].length()));  
//            for (int j = 0; j < imatrix.length; j++) 
//            {  
//                String out = "" + imatrix[i][j];  
//                sb.append(StringTools.fillin(out, 8, false, ' ',8 - out.length()));  
//            }  
//            System.out.println(sb.toString());  
//        }    
//        System.out.println("准确度: " + nf.format(confMatrix.totalAccuracy()));  
//        System.out.println("总共正确数 : " + confMatrix.totalCorrect());  
//        System.out.println("总数：" + confMatrix.totalCount());  
	}

}
