package MLP;

public class MLP_PSO {
	
	
	double[][] input; /* base de treino */
	double[] output; /* saida da bade de traino */
	
	double[][] inputValidate; /* base de validação */
	double[] outputValidate; /* saida da base validação */
	
	int hiddenNeurons; /* Quantidade de neuronios escondidos */
	double learning; /* taxa de aprendisado  */
	
	double[] erroValidate; /* Erro  de Validação */
	
	double[][] population; /* População */
	double[][] velocity;   /* Velocidade de ajuste */
	double[] fitness;      /* Erro quadratico medio*/
	double[][] pBest;      /* Memoria anterior */
	double[] gBest;        /* melhor particula*/                                                 

	public MLP_PSO(double[][] input, double[] output, double[][] inputValidate, double[] outputValidate,
			int hiddenNeurons, double learning, int populationSize) {
		super();
		this.input = input;
		this.output = output;
		this.inputValidate = inputValidate;
		this.outputValidate = outputValidate;
		this.hiddenNeurons = hiddenNeurons;
		this.learning = learning;
		
		int weights = input[0].length * hiddenNeurons + 2 * hiddenNeurons + 1;
		
		this.population = new double[populationSize][weights];
		this.velocity = new double[populationSize][weights];
		this.pBest = new double[populationSize][ weights];
		this.gBest = new double[weights];
		this.fitness = new double[populationSize];
		
		this.generatePopulation();
		
		
	}
	
	public void start (int epooc) {
		for (int i = 0; i < epooc; i++) {
			this.calc_fitness();
			this.populationAjust();
		}
	}

	/* Gerar pupulação, velocidade de ajuste e memoria inicial */	
	public void  generatePopulation () {
		
		for (int i = 0; i < this.population.length; i++) {
			for (int j = 0; j < this.population[0].length; j++) {
				this.population[i][j] = Math.random();
				this.pBest[i][j] = this.population[i][j];
				this.velocity[i][j] = this.population[i][j];
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
		
		
		for (int k = 0; k < this.population.length; k++) {
			
			int p = -1;

			for (int i = 0; i < this.input.length; i++) {
				for (int h = 0; h < net.length; h++) {	
					for (int j = 0; j < this.input[0].length; j++) {
						p++;
						net[h] += this.population[k][p] * this.input[i][j];
						
					}
				}
				
				for (int g = 0; g < net.length; g++) {
					p++;
					net[g] += this.population[k][p];
					
				}

				for (int y = 0; y < net.length; y++) {
					p++;
					netOut += this.population[k][p] * net[y];	
				}

				netOut += this.population[k][p+1];

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
			
			if(this.fitness != null) {
				if(errorTotal < this.fitness[k]) {
					for (int i = 0; i < this.population[0].length; i++) {
						this.pBest[k][i] = this.population[k][i];
					}
					this.fitness[k] = errorTotal;
				}else {
					for (int i = 0; i < this.population[0].length; i++) {
						this.population[k][i] = this.pBest[k][i];
					}
				}
			}else {
				this.fitness[k] = errorTotal;
			}
			
			
		}
		
		/* encontrando melhor particula da população */
		
		double small =  9999999;
		
		for (int i = 0; i < this.fitness.length; i++) {
			
			if(small > this.fitness[i]) {
				
				small = this.fitness[i];
				
				for (int j = 0; j < this.gBest.length; j++) {
					this.gBest[j] = this.population[i][j];
				}
			}
		}

		
	}
	
	/* Metodo de ajuste da população */
	public void populationAjust() {
		
		this.velocityAjust();
		
		for (int i = 0; i < population.length; i++) {
			for (int j = 0; j < population[0].length; j++) {
				this.population[i][j] = this.population[i][j] + this.velocity[i][j];
			}
		}
	}
	/* metodo de calculo da velocidade */
	public void velocityAjust () {
		for (int i = 0; i < this.velocity.length; i++) {
			for (int j = 0; j < this.velocity[0].length; j++) {
				System.out.println("Antes =>"+this.velocity[i][j]);
				this.velocity[i][j] = this.velocity[i][j] + 2 * 
						Math.random() * (this.pBest[i][j] - this.population[i][j]) + 2 * 
						Math.random() * (this.gBest[j] - this.population[i][j]);
				
				System.out.println("Depois =>"+this.velocity[i][j]);
			}
		}
	}
	
	/* Metodo de ativação */
	public double sigmoid (double value) {
		return 1/( 1 + Math.exp(-value));
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


