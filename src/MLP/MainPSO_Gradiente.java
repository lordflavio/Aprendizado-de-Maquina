package MLP;

import javax.swing.JFrame;

import org.math.plot.Plot2DPanel;

public class MainPSO_Gradiente {

	public static void main(String[] args) {
		
		Bases base = new Bases();

		int epooc = 150;
		
		//MLP_PSO_GRADIENTE(base, baseTrain, baseValidade, test, hiddenNeurons, learning, populationSize, c1, c2, window, wInertia, maxInertia, minInertia)
		
		MLP_PSO_GRADIENTE mlp = new MLP_PSO_GRADIENTE(base.getBaseDolar(), 0.30, 0.20, 0.50, 10, 0.01, 50, 2, 2, 1, 0.8, 0.8, 0.2);
		mlp.start(epooc);
		
		
		String s ="";
		
		for (int i = 0; i < mlp.getOutputTest().length; i++) {
		//	s += "Valor Real: "+mlp.getOutputTest()[i] + " | Previs�o PSO: "+mlp.getMlpOutputPSO()[i] + " | Previs�o PSO Gradiente: "+mlp.getMlpOutputPSOGradiente()[i] + "\n"; 
		}
		
		System.out.println(s);


		Plot2DPanel plot = new Plot2DPanel();
		double x[] = new double[epooc];
		for (int i = 0; i < epooc; i++) {
			x[i]=i;
			
			//System.out.println(p.getgBestFitness()[i]);
		}
		
		plot.addLinePlot("PSO", x,mlp.getgBestFitness());
		plot.addLinePlot("GRADIENTE", x,mlp.getErroTotal());
		plot.addLinePlot("GRADIENTE/VALIDA��O", x,mlp.getErroValidate());
		JFrame frame = new JFrame("converg�ncia");
		frame.setContentPane(plot);
		frame.setSize(700, 500);

		frame.setVisible(true);

		Plot2DPanel plot2 = new Plot2DPanel();

		double[] y = new double[mlp.inputTest.length]; 
		for (int i = 0; i < y.length; i++) {
			y[i] = i;
		}
		plot2.addLinePlot("Valores Reais", y,mlp.getOutputTest());
		plot2.addLinePlot("Previs�o PSO", y,mlp.getMlpOutputPSO());
		//plot2.addLinePlot("PREVIS�ES", y,mlp.getMlpPSOGradientePrevision());
		plot2.addLinePlot("Previs�o PSO/Gradiente", y,mlp.getMlpOutputPSOGradiente());
		JFrame frame2 = new JFrame("Previs�es");
		frame2.setContentPane(plot2);
		frame2.setSize(700, 500);

		frame2.setVisible(true);
	}

}
