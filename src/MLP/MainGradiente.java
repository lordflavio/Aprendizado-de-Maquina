package MLP;

import javax.swing.JFrame;

import org.math.plot.Plot2DPanel;

public class MainGradiente {
	
	double[] base;
	double[] baseNormalized;
	
	double baseTrain;
	double baseValidate;
	double baseTest;
	
	double[][] input; /* base de treino  */
	double[] output; /* saida da bade de traino  */
	
	double[][] inputValidate; /* base de validação */
	double[] outputValidate; /* saida da base validação */
	
	double[][] inputTest; /* base de validação */
	double[] outputTest; /* saida da base validação */

	public static void main(String[] args) {
		
		double[][] baseInput = {{0,0},{0,1},{1,0},{1,1}};
		double[] baseOutput = {0,1,1,0};
		
		double[][] baseValidate = {{0,0},{0,1},{1,0},{1,1}};;
		double[] baseOutValidate = {0,1,1,0}; 
	
		double[][] baseTest = {{0,0},{0,1},{1,0},{1,1}};
		double[] baseOutTest = {0,1,1,0};
	
		int epooc = 1000;
		
		Bases base = new Bases();
		
		
		
//	//	MLP_GRADIENTE mlp = new MLP_GRADIENTE(baseInput, baseOutput, baseValidate, baseOutValidate,10, 0.9);
//		  
//		double erro[] = mlp.train(epooc);
//		double[] result = mlp.generateMlp(baseTest, baseOutTest);
//		double[] erroValidate = mlp.getErroValidate();
//		
//		for (int i = 0; i < result.length; i++) {
//			System.out.println("Saidas Desejadas => "+baseOutTest[i]+" Saidas => "+ result[i]);
//		}
//		
//		
//		Plot2DPanel plot = new Plot2DPanel();
//		double x[] = new double[epooc];
//		for (int i = 0; i < epooc; i++) {
//			x[i]=i;
//			
//			//System.out.println(p.getgBestFitness()[i]);
//		}
//		
//		plot.addLinePlot("Treino", x,erro);
//		plot.addLinePlot("Validação", x,erroValidate);
//		 JFrame frame = new JFrame("Plot");
//		  frame.setContentPane(plot);
//		  frame.setSize(1000, 600);
//		  frame.setVisible(true);
//		
//
	}
	
public void createWindow(int window) {
		
		int train =  (int) Math.round(base.length * baseTrain);
		int validade = (int) Math.round(base.length * baseValidate);
		int test = (int) Math.round(base.length * baseValidate);
		
		this.input = new double[train][window];
		this.output = new double[train];
		
		this.inputValidate = new double[validade][window];
		this.outputValidate = new double[validade];
		
		this.inputTest = new double[test][window];
		this.outputTest = new double[test];
		
		int aux = -1;
		
			for (int i = 0; i < this.input.length; i++) {
				aux++;
				for (int j = 0; j <  this.input[0].length; j++) {
					  this.input[i][j] = this.baseNormalized[aux+j];
				}
				
				this.output[i] = baseNormalized[aux+window];
			}
			
			for (int i = 0; i < this.inputValidate.length; i++) {
				aux++;
				for (int k = 0; k < this.inputValidate[0].length; k++) {
					this.inputValidate[i][k] = this.baseNormalized[aux+k];
				}
				this.outputValidate[i] = baseNormalized[aux+window];
			}
			
			for (int i = 0; i < this.inputTest.length; i++) {
				aux++;
				for (int j = 0; j < this.inputTest[0].length; j++) {
					this.inputTest[i][j] = baseNormalized[aux+j];
				}
				
				this.outputTest[i] = base[aux+window];
			}
			
			String s ="", s1="", s2= "";
			
			for (int i = 0; i < this.input.length; i++) {
				for (int j = 0; j < this.input[0].length; j++) {
					s += this.input[i][j] + " | ";
				}
				
				s += "|=>" + this.output[i] + "\n";
			}
			
			for (int i = 0; i < this.inputValidate.length; i++) {
				for (int j = 0; j < this.inputValidate[0].length; j++) {
					s1 += this.inputValidate[i][j] + " | ";
				}
				
				s1 += "|=>" + this.outputValidate[i] + "\n";
			}
			
			for (int i = 0; i < this.inputTest.length; i++) {
				for (int j = 0; j < this.inputTest[0].length; j++) {
					s2 += this.inputTest[i][j] + " | ";
				}
				
				s2 += "|=>" + this.outputTest[i] + "\n";
			}
			
			
//			System.out.println(s);
//			System.out.println("__________________________________________________");
//			System.out.println(s1);
//			System.out.println("__________________________________________________");
//			System.out.println(s2);
			
	}

}
