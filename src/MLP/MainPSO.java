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

public class MainPSO {
	
	public static void main(String[] args) throws IOException {
		
		Bases base = new Bases();

		int janela = 3;
		
		double[][] baseInput = new double[base.getBase().length - janela][janela];
		double[] baseOutput = new double[base.getBase().length  - janela];
		
		for (int i = 0; i < base.getBase().length  - janela; i++) {
			for (int j = 0; j < baseInput[0].length; j++) {
				baseInput[i][j] = base.getBase()[i+j];
			}
			baseOutput[i] = base.getBase()[i+janela];
		}
		
		int epooc = 100;
		
	    //new MLP_PSO(base, hiddenNeurons, learning, populationSize, c1, c2, window, wInertia, maxInertia, minInertia)
		MLP_PSO p = new MLP_PSO(base.getBase(),10, 0.5, 50,2,2,janela,0.8,0.8,0.2);

		p.start(epooc);
		p.test(baseInput, baseOutput);
	
		Plot2DPanel plot = new Plot2DPanel();
		double x[] = new double[epooc];
		for (int i = 0; i < p.getgBestFitness().length; i++) {
			x[i]=i;
			
			//System.out.println(p.getgBestFitness()[i]);
		}
		
		plot.addLinePlot("Treino", x,p.getgBestFitness());
		//plot.addLinePlot("Validação", x,erroValidate);
		JFrame frame = new JFrame("Plot");
		frame.setContentPane(plot);
		frame.setSize(700, 500);

		frame.setVisible(true);

		Plot2DPanel plot2 = new Plot2DPanel();

		double[] y = new double[baseInput.length]; 
		for (int i = 0; i < baseInput.length; i++) {
			y[i] = i;
		}
		plot2.addLinePlot("Saidas", y,baseOutput);
		plot2.addLinePlot("Previsão", y,p.getMlpOutput());
		//plot.addLinePlot("Validação", x,erroValidate);
		JFrame frame2 = new JFrame("Plot2");
		frame2.setContentPane(plot2);
		frame2.setSize(700, 500);

		frame2.setVisible(true);
//		
//
//
	}

}
