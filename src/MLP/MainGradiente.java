package MLP;

import javax.swing.JFrame;

import org.math.plot.Plot2DPanel;

public class MainGradiente {

	public static void main(String[] args) {
		
		double[][] baseInput = {{0,0},{0,1},{1,0},{1,1}};
		double[] baseOutput = {0,1,1,0};
		
		double[][] baseValidate = {{0,0},{0,1},{1,0},{1,1}};;
		double[] baseOutValidate = {0,1,1,0}; 
	
		double[][] baseTest = {{0,0},{0,1},{1,0},{1,1}};
		double[] baseOutTest = {0,1,1,0};
	
		int epooc = 1000;
		
		MLP_GRADIENTE mlp = new MLP_GRADIENTE(baseInput, baseOutput, baseValidate, baseOutValidate,10, 0.9);
		  
		double erro[] = mlp.train(epooc);
		double[] result = mlp.generateMlp(baseTest, baseOutTest);
		double[] erroValidate = mlp.getErroValidate();
		
		for (int i = 0; i < result.length; i++) {
			System.out.println("Saidas Desejadas => "+baseOutTest[i]+" Saidas => "+ result[i]);
		}
		
		
		Plot2DPanel plot = new Plot2DPanel();
		double x[] = new double[epooc];
		for (int i = 0; i < epooc; i++) {
			x[i]=i;
			
			//System.out.println(p.getgBestFitness()[i]);
		}
		
		plot.addLinePlot("Treino", x,erro);
		plot.addLinePlot("Validação", x,erroValidate);
		 JFrame frame = new JFrame("Plot");
		  frame.setContentPane(plot);
		  frame.setSize(1000, 600);
		  frame.setVisible(true);
		

	}

}
