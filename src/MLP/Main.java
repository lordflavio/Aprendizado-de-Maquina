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
	
	public static void main(String[] args) throws IOException {

		
		double[][] baseInput = {{0,0},{0,1},{1,0},{1,1}};
		double[] baseOutput = {0,1,1,0};
		
		double[][] baseValidate = {{0,0},{0,1},{1,0},{1,1}};;
		double[] baseOutValidate = {0,1,1,0}; 
	
		double[][] baseTest = {{0,0},{0,1},{1,0},{1,1}};
		double[] baseOutTest = {0,1,1,0};
	
		int epooc = 100;
		
		
		
		MLP_PSO p = new MLP_PSO(baseInput, baseOutput, baseValidate, baseOutValidate, 2, 0.5, 50,2,2);
		
		p.start(epooc);
		p.test(baseTest, baseOutTest);
		
		
		
//		Mlp mlp = new Mlp(baseInput, baseOutput, baseValidate, baseOutValidate,3, 0.8);
//		  
//		double erro[] = mlp.train(epoca);
//		double[] result = mlp.generateMlp(baseTest, baseOutTest);
//		double[] erroValidate = mlp.getErroValidate();
//		
//		for (int i = 0; i < result.length; i++) {
//			System.out.println("Saidas Desejadas => "+baseOutTest[i]+" Saidas => "+ result[i]);
//		}
//		
		Plot2DPanel plot = new Plot2DPanel();
		double x[] = new double[epooc];
		for (int i = 0; i < p.getgBestFitness().length; i++) {
			x[i]=i;
			
			System.out.println(p.getgBestFitness()[i]);
		}
		
		plot.addLinePlot("Treino", x,p.getgBestFitness());
		//plot.addLinePlot("Validação", x,erroValidate);
		 JFrame frame = new JFrame("Plot");
		  frame.setContentPane(plot);
		  frame.setSize(300, 300);
		  
		  frame.setVisible(true);
//		
//
//
	}

}
