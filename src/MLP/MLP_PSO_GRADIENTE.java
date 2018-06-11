package MLP;

public class MLP_PSO_GRADIENTE {
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
	
	int hiddenNeurons; /* Quantidade de neuronios escondidos */
	double learning; /* taxa de aprendisado  */
	
	double[] erroValidate; /* Erro  de Validação */
	
	double[][] inputWeights; /* pesos do treino */
	double[][] outputWeights; /* peso da saida do treino */
	
	double[] biasInputWeights; /* peso do bias */
	double[] biasOuputWeights; /* peso saida bias */
	
	double[] erroTotal;
	
	double[][] particle;		 /* População */
	double[][] velocity;   		/* Velocidade de ajuste */
	double[] particleFitness;      /* Erro quadratico medio*/
	double[][] pBest;      		/* Memoria anterior */
	double[] pBestFitness;
	double[] gBest;        		/* melhor particula*/
	double[] gBestFitness;
	double c1,c2;
	double min,max;
	
	double wInertia;
	double maxInertia;
	double minInertia;
	
	double[] mlpOutputPSO;           /* Saidas da rede*/
	double[] mlpOutputPSOGradiente; /* Saidas da rede*/

	public MLP_PSO_GRADIENTE(double[] base, double baseTrain, double baseValidade, double baseTest, int hiddenNeurons, double learning, int populationSize, double c1, double c2, 
				   int window, double wInertia, double maxInertia,double minInertia) {
		super();
		
		this.base = base;
		this.baseTrain = baseTrain;
		this.baseValidate = baseValidade;
		this.baseTest = baseTest;
		this.hiddenNeurons = hiddenNeurons;
		this.learning = learning;
		this.c1 = c1;
		this.c2 = c2;
		this.wInertia = wInertia;
		this.maxInertia = maxInertia;
		this.minInertia = minInertia;
		
		this.baseNormalized = new double[base.length];
		
		this.normalize();
		this.createWindow(window);

		int weights = input[0].length * hiddenNeurons + 2 * hiddenNeurons + 1;
		
		this.particle = new double[populationSize][weights];
		this.velocity = new double[populationSize][weights];
		this.pBest = new double[populationSize][ weights];
		this.gBest = new double[weights];
		this.particleFitness = new double[populationSize];
		this.pBestFitness = new double[populationSize];
		
		this.inputWeights = new double[input[0].length][hiddenNeurons];
		this.biasInputWeights = new double[hiddenNeurons];
		
		this.outputWeights = new double[hiddenNeurons][1];
		this.biasOuputWeights = new double[1];
		
		this.generatePopulation();
	}
	
	public void start (int epooc) {
		this.gBestFitness = new double[epooc];
		for (int i = 0; i < epooc; i++) {
			this.calc_fitness();
			this.calc_gBest(i);
			this.populationAjust();
			this.inertiaAjust(i, epooc);
		}
		
		this.generateWeights();
		this.erroTotal = this.train(epooc);
		
	}
	
	/* ----------------------------------------------------- PSO------------------------------------------------------- */
	
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
		
		double[] net = new double[this.hiddenNeurons];
		double netOut = 0;
		
		double error;
		double errorTotal = 0;

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

				//netOut = sigmoid(netOut);

				error = (this.output[i] - netOut);
				
				errorTotal += Math.pow(error, 2); /*eé aqui meu filhoooooo*/
				
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
			
			errorTotal = 0;
		}
	}

	public void calc_gBest (int index) {
	/* encontrando melhor particula da população */
		
		int i = this.minFitness(pBestFitness);
		
		if(index != 0) {

			if(gBestFitness[index-1] > this.pBestFitness[i]) {

				gBestFitness[index] = pBestFitness[i];
				
				for (int j = 0; j < this.gBest.length; j++) {
					this.gBest[j] = this.pBest[i][j];
				}
					
			}else {
				gBestFitness[index] = gBestFitness[index-1];
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
				this.velocity[i][j] = this.wInertia * this.velocity[i][j] + this.c1 * 
						Math.random() * (this.pBest[i][j] - this.particle[i][j]) + this.c2 * 
						Math.random() * (this.gBest[j] - this.particle[i][j]);
				
				//System.out.println("Depois =>"+this.velocity[i][j]);
			}
		}
	}
	
	public void inertiaAjust(int index, int epooc) {
		this.wInertia = this.maxInertia - index / epooc * (this.maxInertia - this.minInertia);
	}
	
	/* ----------------------------------------------------- GRADIENTE------------------------------------------------------- */
	
	/* Gegar pesos aleatorios para arrays[][] */
	public void generateWeights() {
		int p = -1;
		/*pesos da entrada*/
		for (int i = 0; i < this.inputWeights.length; i++) {
			for (int j = 0; j < this.inputWeights[0].length; j++) {
				p++;
				this.inputWeights[i][j] = this.gBest[p];
			}
		}
		
		for (int i = 0; i < this.outputWeights.length; i++) {
			for (int j = 0; j < this.outputWeights[0].length; j++) {
				p++;
				this.outputWeights[i][j] = this.gBest[p];
			}
		}
		
		for (int i = 0; i < this.biasInputWeights.length; i++) {
			p++;
			this.biasInputWeights[i] = this.gBest[p];
		}
		
		this.biasOuputWeights[0] = this.gBest[p+1];

	}
	
	/* Metodo de Treino */
	public double[] train (int epoca) {
		
		this.erroValidate = new double[epoca];
		
		double[] net = new double[this.hiddenNeurons];
		double netOut = 0;
		
		double erro = 0;
		
		double[] erroTotal = new double[epoca];  
		
		double[] gradients = new double[this.hiddenNeurons]; 
		double gradientOut = 0;
		
		for (int n = 0; n < epoca; n++) { /* Número de interações */
			
			for (int i = 0; i < input.length; i++) { /* Linhas da Base  */
				
				for (int j = 0; j < this.hiddenNeurons; j++) { /* Neuronios escondiodos */
					
					for (int k = 0; k < input[0].length; k++) { /* coulunas da base*/
						
						net[j] +=  this.inputWeights[k][j] * this.input[i][k];
					}
					
						net[j] += this.biasInputWeights[j];
															
						net[j] = sigmoid(net[j]);
				}
				
				
				for (int j = 0; j < this.hiddenNeurons; j++) {
					 
						netOut += this.outputWeights[j][0] * net[j];
					   
				}
				
				netOut += this.biasOuputWeights[0];
				
			//netOut = sigmoid(netOut);
				
				//System.out.println("Saida desejada: =>"+this.output[i] +" | Saida Obtida =>" + netOut);
				
				erro = (this.output[i] - netOut);
				
				erroTotal[n] += Math.pow(erro, 2);
				
				
				gradientOut = erro * netOut * (1 - netOut); /*calculo do gradiente do neurônio de saida */
				
				for (int j = 0; j < gradients.length; j++) { /*calculo do gradiente dos neurônio escondidos */
					gradients[j] = this.outputWeights[j][0] * gradientOut;
					gradients[j] = gradients[j] * net[j] * (1 - net[j]);
				}
				
				/* Ajuste de pesos */
				
				for (int j = 0; j < this.hiddenNeurons; j++) {
					
					this.outputWeights[j][0] += this.learning * net[j] * gradientOut;
					
					for (int g = 0; g < this.input[0].length; g++) {
						this.inputWeights[g][j] += this.learning * this.input[i][g] * gradients[j]; 
					}
				}
				
				/* Ajuste de Bias */
				for (int j = 0; j < this.biasOuputWeights.length; j++) {
					this.biasOuputWeights[j] += this.learning * 1 * gradientOut;
				}
				
				for (int j = 0; j < this.hiddenNeurons; j++) {
					this.biasInputWeights[j] += this.learning * 1 * gradients[j];
					
				}
				
				/* Set de Variaveis */
				
				netOut = 0;
				gradientOut = 0;
				
				for (int j = 0; j < this.hiddenNeurons; j++) {
					
					net[j] = 0;
					gradients[j] = 0;
				}

				//System.out.println("------------------------------------------------------------");

			}
			
			
			this.erroValidate[n] = this.validate(this.inputValidate, this.outputValidate);
			
			erroTotal[n] = erroTotal[n] / this.input.length;
			
		}
		
		return erroTotal;
	}
	
	/* Metodo de Validação */
	public double validate(double[][] inputValidate, double[] outputValidate) {
		double[] net = new double[this.hiddenNeurons];
		double netOut = 0;
		double erro = 0;
        double erroTotal = 0;  
		double[] gradients = new double[this.hiddenNeurons]; 
		double gradientOut = 0;
		
		for (int i = 0; i < inputValidate.length; i++) {
			for (int j = 0; j < this.hiddenNeurons; j++) {
				for (int k = 0; k < inputValidate[0].length; k++) {
					net[j] +=  this.inputWeights[k][j] * inputValidate[i][k];
				}
				
				net[j] += this.biasInputWeights[j];
				
				net[j] = sigmoid(net[j]);
			}
			
			for (int j = 0; j < this.hiddenNeurons; j++) {  
				netOut += this.outputWeights[j][0] * net[j];
			}
			
			netOut += this.biasOuputWeights[0];	 
			
			netOut = sigmoid(netOut);
			
			erro = (outputValidate[i] - netOut);
			
			erroTotal += Math.pow(erro, 2);
			
			netOut = 0;
			
			for (int j = 0; j < this.hiddenNeurons; j++) {
				net[j] = 0;
			}
			
		}
		
		return (erroTotal/inputValidate.length);
	}
	
	/* Metodo de ativação */
	public double sigmoid (double value) {
		return 1/( 1 + Math.exp(-value));
	}
	
	public int minFitness (double[] value) {
		
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
		
	public void normalize () {
		
		this.max = 0;
		this.min = base[0];
		
		for (int i = 0; i < base.length; i++) {
			 
			 if(this.max < base[i]) {
				 this.max = base[i];
			 }
			 
			 if(this.min > base[i]) {
				 this.min = base[i];
			 }
		}
		
		for (int i = 0; i < base.length; i++) {
			this.baseNormalized[i] = (this.base[i] - this.max) / (this.max - this.min);
		}
	}
	
	public double  denormalize (double value) {
		return value * (this.max - this.min) + this.max;
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
			
	}
	
	public double[] getgBestFitness() {
		return gBestFitness;
	}
	
	public double[] getMlpOutputPSO() {
		return mlpOutputPSO;
	}

	public double[] getMlpOutputPSOGradiente() {
		return mlpOutputPSOGradiente;
	}

	public double[] getErroValidate() {
		return erroValidate;
	}
	
	public double[] getErroTotal() {
		return erroTotal;
	}

	public double[] getOutputTest() {
		return outputTest;
	}

	public void generateMlpPSO () {
		
		this.mlpOutputPSO = new double[input.length];
		double[] net = new double[this.hiddenNeurons];
		double netOut = 0;
		
		double error;
		double errorTotal = 0;
		
		int p = -1;

		for (int i = 0; i < this.inputTest.length; i++) {
			for (int h = 0; h < net.length; h++) {	
				for (int j = 0; j < this.inputTest[0].length; j++) {
					p++;
					net[h] += this.gBest[p] * this.inputTest[i][j];
					
				}
			}
			
			for (int g = 0; g < net.length; g++) {
				p++;
				net[g] += this.gBest[p];
				
				net[g] = this.sigmoid(net[g]);
				
			}

			for (int y = 0; y < net.length; y++) {
				p++;
				netOut += this.gBest[p] * net[y];	
			}

			netOut += this.gBest[p+1];

			//netOut = this.sigmoid(netOut);
			this.mlpOutputPSO[i] = this.denormalize(netOut);
			//System.out.println("Saida Esperada => "+output[i]+" | Saida da rede => "+this.denormalize(netOut));

			netOut = 0;
			p = -1;
			
			for (int j = 0; j < this.hiddenNeurons; j++) {
				net[j] = 0;
			}

		}
	}

	public void generateMlpPSOGradiente (){
		
		this.mlpOutputPSOGradiente = new double[inputTest.length];
		
		double[] net = new double[this.hiddenNeurons];
		double netOut = 0;
		double erro = 0;
        double erroTotal = 0;  
		double[] gradients = new double[this.hiddenNeurons]; 
		double gradientOut = 0;
		
		for (int i = 0; i < this.inputTest.length; i++) {
			for (int j = 0; j < this.hiddenNeurons; j++) {
				for (int k = 0; k < this.inputTest[0].length; k++) {
					net[j] +=  this.inputWeights[k][j] * this.inputTest[i][k];
				}
				
				net[j] += this.biasInputWeights[j];
				
				net[j] = sigmoid(net[j]);
			}
			
			for (int j = 0; j < this.hiddenNeurons; j++) {  
				netOut += this.outputWeights[j][0] * net[j];
			}
			
			netOut += this.biasOuputWeights[0];	 
			
			//netOut = sigmoid(netOut);
			
			this.mlpOutputPSOGradiente[i] = this.denormalize(netOut);
			
//			erro = (outputTest[i] - netOut);
//			
//			erroTotal += Math.pow(erro, 2);
			
			netOut = 0;
			
			for (int j = 0; j < this.hiddenNeurons; j++) {
				net[j] = 0;
			}
			
		}
	}
	
}
