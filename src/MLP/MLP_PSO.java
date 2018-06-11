package MLP;

public class MLP_PSO {
	
	double[] base;
	double[] baseNormalized;

	double[][] input; /* base de treino  */
	double[] output; /* saida da bade de traino  */
	
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
	double[] gBest;        		/* melhor particula*/
	double[] gBestFitness;
	double c1,c2;
	double min,max;
	
	double wInertia;
	double maxInertia;
	double minInertia;
	
	double[] mlpOutput; /* Saidas da rede*/

	public MLP_PSO(double[] base, int hiddenNeurons, double learning, int populationSize, double c1, double c2, 
				   int window, double wInertia, double maxInertia,double minInertia) {
		super();
		
		this.base = base;
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
		
		this.input = new double[base.length][window];
		this.output = new double[base.length];
		
		for (int i = 0; i < base.length - 2; i++) {
			for (int j = 0; j < this.input[0].length; j++) {
				this.input[i][j] = this.baseNormalized[i+j];
			}
			this.output[i] = baseNormalized[i+2];
		}
	}
	
	public void inertiaAjust(int index, int epooc) {
		this.wInertia = this.maxInertia - index / epooc * (this.maxInertia - this.minInertia);
	}
	
	public double[] getgBestFitness() {
		return gBestFitness;
	}	
	
	public double[] getMlpOutput() {
		return mlpOutput;
	}

	public void test (double[][] input, double[] output) {
		
		this.mlpOutput = new double[input.length];
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
				
				net[g] = this.sigmoid(net[g]);
				
			}

			for (int y = 0; y < net.length; y++) {
				p++;
				netOut += this.gBest[p] * net[y];	
			}

			netOut += this.gBest[p+1];

			//netOut = this.sigmoid(netOut);
			this.mlpOutput[i] = this.denormalize(netOut);
			//System.out.println("Saida Esperada => "+output[i]+" | Saida da rede => "+this.denormalize(netOut));

			netOut = 0;
			p = -1;
			
			for (int j = 0; j < this.hiddenNeurons; j++) {
				net[j] = 0;
			}

		}
	}

}


