package MLP;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.LineNumberReader;

import javax.swing.JFrame;

import org.math.plot.Plot2DPanel;

public class MainArquivoBase {

	public static void main(String[] args) throws IOException {
		MainArquivoBase a = new MainArquivoBase();
		double[] baseI = a.mountBase("C:\\Users\\Flavio\\eclipse-workspace\\Aprendisado_de_Maquina\\bin\\Arquivos\\carsales.txt");
		double[] baseV = a.mountBase("C:\\Users\\Flavio\\eclipse-workspace\\Aprendisado_de_Maquina\\bin\\Arquivos\\carsalesT.txt");
		double[] baseT = a.mountBase("C:\\Users\\Flavio\\eclipse-workspace\\Aprendisado_de_Maquina\\bin\\Arquivos\\carsalesV.txt");
		
		
		int janela = 2;
		
		double[][] baseInput = new double[baseI.length - janela][janela];
		double[] baseOutput = new double[baseI.length - janela];
		
		double[][] baseValidate = new double[baseV.length - janela][janela];
		double[] baseOutValidate =  new double[baseV.length - janela];
		
		double[][] baseTest = new double[baseT.length - janela][janela];
		double[] baseOutTest = new double[baseT.length - janela];
		
		
		for (int i = 0; i < baseV.length - 2; i++) {
			for (int j = 0; j < baseValidate[0].length; j++) {
				baseValidate[i][j] = baseV[i+j];
			}
			baseOutValidate[i] = baseV[i+2];
		}
		
		
		for (int i = 0; i < baseI.length - 2; i++) {
			for (int j = 0; j < baseInput[0].length; j++) {
				baseInput[i][j] = baseI[i+j];
			}
				baseOutput[i] = baseI[i+2];
		}
		
		
		for (int i = 0; i < baseT.length - 2; i++) {
			for (int j = 0; j < baseTest[0].length; j++) {
				baseTest[i][j] = baseT[i+j];
			}
			    baseOutTest[i] = baseT[i+2];
		}
		
		
		
		
//		for (int i = 0; i < baseOutput.length; i++) {
//			System.out.println(baseOutput[i]);
//		}
//		
//		String s = "";
//		
//		for (int i = 0; i < baseInput.length; i++) {
//			for (int j = 0; j < baseInput[0].length; j++) {
//				s+= baseInput[i][j]+" | ";
//			}
//			System.out.println(s);
//			s="";
//		}
		
		
	
		int epoca = 2000;
		
		MLP_GRADIENTE mlp = new MLP_GRADIENTE(baseInput, baseOutput, baseValidate, baseOutValidate,3, 0.95);
		  
		double erro[] = mlp.train(epoca);
		double[] result = mlp.generateMlp(baseTest, baseOutTest);
		double[] erroValidate = mlp.getErroValidate();
		
//		for (int i = 0; i < result.length; i++) {
//			System.out.println("Saidas "+ result[i]);
//		}
		
//		Plot2DPanel plot = new Plot2DPanel();
//		double x[] = new double[epoca];
//		for (int i = 0; i < erro.length; i++) {
//			x[i]=i;
//		}
		
//		plot.addLinePlot("Treino", x,erro);
//		plot.addLinePlot("Validação", x,erroValidate);
//		 JFrame frame = new JFrame("Plot");
//		  frame.setContentPane(plot);
//		  frame.setSize(300, 300);
//		  
//		  frame.setVisible(true);
		


	}
	
	public double[] mountBase (String url) throws IOException{

		File arquivo = new File(url);
		LineNumberReader linhaLeitura = new LineNumberReader(new FileReader(arquivo));
		linhaLeitura.skip(arquivo.length());
		int qtdLinha = linhaLeitura.getLineNumber();
		
		//System.out.println(qtdLinha);

		double[] base = new double[qtdLinha + 1];
		
		int i = 0;		
		
		BufferedReader br = new BufferedReader(new FileReader(arquivo));
		while(br.ready()){
		   String linha = br.readLine();
		 //  System.out.println(linha);
		   
		  int n = Integer.parseInt(linha);

		  if( i < qtdLinha) {
			  base[i] = n;
			  i++; 
		  }
		  
		}
		br.close();
		
		return base;
	}
}
