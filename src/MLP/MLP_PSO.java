package MLP;

public class MLP_PSO {
	
	
	double[][] input; /* base de treino */
	double[] output; /* saida da bade de traino */
	
	double[][] inputValidate; /* base de validação */
	double[] outputValidate; /* saida da base validação */
	
	int hiddenNeurons; /* Quantidade de neuronios escondidos */
	double learning; /* taxa de aprendisado  */
	
	double[] erroValidate; /* Erro  de Validação */
	
	double[][] particle;		 /* População */
	double[][] velocity;   		/* Velocidade de ajuste */
	double[] particleFitness;      /* Erro quadratico medio*/
	double[][] pBest;      		/* Memoria anterior */
	double[] pBestFitness;
	double[] gBest;        			/* melhor particula*/
	double[] gBestFitness;
	double c1,c2;

	public MLP_PSO(double[][] input, double[] output, double[][] inputValidate, double[] outputValidate,
			int hiddenNeurons, double learning, int populationSize, double c1, double c2) {
		super();
		this.input = input;
		this.output = output;
		this.inputValidate = inputValidate;
		this.outputValidate = outputValidate;
		this.hiddenNeurons = hiddenNeurons;
		this.learning = learning;
		this.c1 = c1;
		this.c2 = c2;
		
		
		int weights = input[0].length * hiddenNeurons + 2 * hiddenNeurons + 1;
		
		this.particle = new double[populationSize][weights];
		this.velocity = new double[populationSize][weights];
		this.pBest = new double[populationSize][ weights];
		this.gBest = new double[weights];
		this.particleFitness = new double[populationSize];
		this.pBestFitness = new double[populationSize];
		
		this.generatePopulation();
		
		
	}
	
	public void start (int epooc) {
		this.gBestFitness = new double[epooc];
		for (int i = 0; i < epooc; i++) {
			this.calc_fitness();
			this.calc_gBest(i);
			this.populationAjust();
		}
	}

	/* Gerar pupulação, velocidade de ajuste e memoria inicial */	
	public void  generatePopulation () {
		
		for (int i = 0; i < this.particle.length; i++) {
			for (int j = 0; j < this.particle[0].length; j++) {
				this.particle[i][j] = Math.random();
				this.pBest[i][j] = this.particle[i][j];
				this.velocity[i][j] = this.particle[i][j];
			}
		}
	}
	
	/* calculando melhor particula a partir do erro quadratico medio */
	public void calc_fitness() {
		
//		int inputWeights = input[0].length * this.hiddenNeurons;
//		int biasInput = this.hiddenNeurons;
//		int outWeights = this.hiddenNeurons;
		
		double[] net = new double[this.hiddenNeurons];
		double netOut = 0;
		
		double error;
		double errorTotal = 0;
		
//		double[][] auxInputWeights = new double[input[0].length][this.hiddenNeurons];
//		double[][] auxOutWeights = new double[this.hiddenNeurons][1];
//		double[] biasInputWeights = new double[this.hiddenNeurons];
//		double[] biasOutputWeights = new double[1];
		
		
		for (int k = 0; k < this.particle.length; k++) {
			
			int p = -1;

			for (int i = 0; i < this.input.length; i++) {
				for (int h = 0; h < net.length; h++) {	
					for (int j = 0; j < this.input[0].length; j++) {
						p++;
						net[h] += this.particle[k][p] * this.input[i][j];
						
					}
				}
				
				for (int g = 0; g < net.length; g++) {
					p++;
					net[g] += this.particle[k][p];
					
					net[g] = sigmoid(net[g]);
					
				}
				

				for (int y = 0; y < net.length; y++) {
					p++;
					netOut += this.particle[k][p] * net[y];	
				}

				netOut += this.particle[k][p+1];

				netOut = sigmoid(netOut);

				error = (this.output[i] - netOut);
				
				errorTotal += Math.pow(error, 2);
				
				error = 0;
				netOut = 0;
				p = -1;
				
				for (int j = 0; j < this.hiddenNeurons; j++) {
					net[j] = 0;
				}

			}
			
			errorTotal = errorTotal / this.input.length;
			
			/* criando / modificando memoria anterior das particulas */
			
			if(this.particleFitness[k] > 0) {
				if(errorTotal < this.particleFitness[k]) {
					for (int i = 0; i < this.particle[0].length; i++) {
						this.pBest[k][i] = this.particle[k][i];
					}
					this.particleFitness[k] = errorTotal;
					this.pBestFitness[k] = errorTotal;
				}else {
					this.particleFitness[k] = errorTotal;
				}
			}else {
				this.particleFitness[k] = errorTotal;
				this.pBestFitness[k] = errorTotal;
			}
			
			
		}
	}

	public void calc_gBest (int index) {
	/* encontrando melhor particula da população */
		
		int i = this.min(pBestFitness);
		
		if(index != 0) {

			if(gBestFitness[index - 1] > this.pBestFitness[i]) {

				gBestFitness[index] = pBestFitness[i];
				
				for (int j = 0; j < this.gBest.length; j++) {
					this.gBest[j] = this.pBest[i][j];
				}
					
			}else {
				gBestFitness[index] = pBestFitness[i];
			}
		}else {
			
			gBestFitness[index] = pBestFitness[i];
			
			for (int j = 0; j < this.gBest.length; j++) {
				this.gBest[j] = this.pBest[i][j];
			}
		}
		
		
	}
	
	/* Metodo de ajuste da população */
	public void populationAjust() {
		
		this.velocityAjust();
		
		for (int i = 0; i < particle.length; i++) {
			for (int j = 0; j < particle[0].length; j++) {
				this.particle[i][j] = this.particle[i][j] + this.velocity[i][j];
			}
		}
	}
	/* metodo de calculo da velocidade */
	public void velocityAjust () {
		for (int i = 0; i < this.velocity.length; i++) {
			for (int j = 0; j < this.velocity[0].length; j++) {
				//System.out.println("Antes =>"+this.velocity[i][j]);
				this.velocity[i][j] = this.velocity[i][j] + this.c1 * 
						Math.random() * (this.pBest[i][j] - this.particle[i][j]) + this.c2 * 
						Math.random() * (this.gBest[j] - this.particle[i][j]);
				
				//System.out.println("Depois =>"+this.velocity[i][j]);
			}
		}
	}
	
	/* Metodo de ativação */
	public double sigmoid (double value) {
		return 1/( 1 + Math.exp(-value));
	}
	
	public int min (double[] value) {
		
		double min = 999999;
		int index = 0;
		
		for (int i = 0; i < value.length; i++) {
			
			if(min > value[i] ) {
				min = value[i];
                index = i;
			}
		}
		
		return index;
	}
	
	public double[] getgBestFitness() {
		return gBestFitness;
	}	
	
	public void test (double[][] input, double[] output) {
		
		double[] net = new double[this.hiddenNeurons];
		double netOut = 0;
		
		double error;
		double errorTotal = 0;
		
		int p = -1;

		for (int i = 0; i < input.length; i++) {
			for (int h = 0; h < net.length; h++) {	
				for (int j = 0; j < this.input[0].length; j++) {
					p++;
					net[h] += this.gBest[p] * this.input[i][j];
					
				}
			}
			
			for (int g = 0; g < net.length; g++) {
				p++;
				net[g] += this.gBest[p];
				
				net[g] = sigmoid(net[g]);
				
			}

			for (int y = 0; y < net.length; y++) {
				p++;
				netOut += this.gBest[p] * net[y];	
			}

			netOut += this.gBest[p+1];

			netOut = sigmoid(netOut);
			
			System.out.println("Saida Esperada => "+output[i]+" | Saida da rede => "+netOut);

			netOut = 0;
			p = -1;
			
			for (int j = 0; j < this.hiddenNeurons; j++) {
				net[j] = 0;
			}

		}
	}

}


