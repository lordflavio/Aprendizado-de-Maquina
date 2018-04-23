package MLP;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.LineNumberReader;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;

import javax.swing.JFrame;

import org.math.plot.Plot2DPanel;

public class Main {
	
	static double[][] baseInput;
	static double[] baseOutput;
	
	static double[][] baseValidate;
	static double[] baseOutValidate;
	
//	double[][] baseTest = {{0,0},{0,1},{1,0},{1,1}};
//	double[] baseOutTest = {0,1,1,0};

	public static void main(String[] args) throws IOException {
		
		Main a = new Main();
		a.mountBase(2);
		a.mountBaseValidate(2);
		
//		double[][] baseInput = {{0,0.1},{0.1,0.1},{0.2,0.1},{0.3,0.1},{0.4,0.1}};;
//		double[] baseOutput = {0.01,0.04,0.09,0.16,0.25};
//		
//		double[][] baseValidate = {{0.5,0.1},{0.6,0.1},{0.7,0.1}};
//		double[] baseOutValidate = {0.36,0.49,0.64};
//		
		double[][] baseTest = {{0,0},{0,1},{1,0},{1,1}};
		double[] baseOutTest = {0,1,1,0};
		
		int epoca = 200;
		
		Mlp mlp = new Mlp(baseInput, baseOutput, baseValidate, baseOutput, baseTest, baseOutTest, 4, 0.1);
		
		double erro[] = mlp.train(epoca);
		double[] erroValidate = mlp.getErroValidate();
		
		Plot2DPanel plot = new Plot2DPanel();
		double x[] = new double[epoca];
		for (int i = 0; i < erro.length; i++) {
			x[i]=i;
		}
		
		plot.addLinePlot("Treino", x,erro);
		plot.addLinePlot("Validação", x,erroValidate);
		 JFrame frame = new JFrame("Plot");
		  frame.setContentPane(plot);
		  frame.setSize(300, 300);
		  
		  frame.setVisible(true);
		
		System.out.println(baseInput.length);
		System.out.println(baseOutput.length);
		
		System.out.println(baseValidate.length);
		System.out.println(baseOutValidate.length);
		
		
	}
	
	public void mountBase (int sizeWin ) throws IOException{

		File arquivo = new File("C:\\Users\\Flavio\\eclipse-workspace\\Aprendisado_de_Maquina\\bin\\Arquivos\\carsales.txt");
		LineNumberReader linhaLeitura = new LineNumberReader(new FileReader(arquivo));
		linhaLeitura.skip(arquivo.length());
		int qtdLinha = linhaLeitura.getLineNumber();
		
		this.baseInput = new double[qtdLinha][sizeWin];
		this.baseOutput = new double[qtdLinha];
		
		double[] base = new double[qtdLinha];
		
		int i = 0;

		
		BufferedReader br = new BufferedReader(new FileReader(arquivo));
		while(br.ready()){
		   String linha = br.readLine();
		 //  System.out.println(linha);
		   
		  int n = Integer.parseInt(linha);
		  
		  base[i] = n;
		  
		  i++;

		}
		br.close();
		
		for (int j = 0; j < baseInput.length; j++) {
			for (int k = 0; k < baseInput[0].length; k++) {
				baseInput[j][k] = base[i+j];
			}
			    baseOutput[j] = base[i+sizeWin];
		}

	}
	
	
	public void mountBaseValidate (int sizeWin ) throws IOException{

		File arquivo = new File("C:\\Users\\Flavio\\eclipse-workspace\\Aprendisado_de_Maquina\\bin\\Arquivos\\carsalesV.txt");
		LineNumberReader linhaLeitura = new LineNumberReader(new FileReader(arquivo));
		linhaLeitura.skip(arquivo.length());
		int qtdLinha = linhaLeitura.getLineNumber();

		this.baseValidate = new double[qtdLinha+1][sizeWin];
		this.baseOutValidate = new double[(qtdLinha+1)/sizeWin];
		
        double[] base = new double[qtdLinha];
		
		int i = 0;

		
		BufferedReader br = new BufferedReader(new FileReader(arquivo));
		while(br.ready()){
		   String linha = br.readLine();
		 //  System.out.println(linha);
		   
		  int n = Integer.parseInt(linha);
		  
		  base[i] = n;
		  
		  i++;

		}
		br.close();
		
		for (int j = 0; j < baseInput.length; j++) {
			for (int k = 0; k < baseInput[0].length; k++) {
				baseValidate[j][k] = base[i+j];
			}
			    baseOutValidate[j] = base[i+sizeWin];
		}

	}
	

}
