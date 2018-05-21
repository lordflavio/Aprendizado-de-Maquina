package MLP;

public class MLP_PSO {
	
	
	double[][] input; /* base de treino */
	double[] output; /* saida da bade de traino */
	
	double[][] inputValidate; /* base de validação */
	double[] outputValidate; /* saida da base validação */
	
	int hiddenNeurons; /* Quantidade de neuronios escondidos */
	double learning; /* taxa de aprendisado  */
	
	double[] erroValidate; /* Erro  de Validação */
	
	double[][] population;
	double[] fitness;
	double[][] pBest;
	double[] gBest;
	

	public MLP_PSO(double[][] input, double[] output, double[][] inputValidate, double[] outputValidate,
			int hiddenNeurons, double learning, int populationSize) {
		super();
		this.input = input;
		this.output = output;
		this.inputValidate = inputValidate;
		this.outputValidate = outputValidate;
		this.hiddenNeurons = hiddenNeurons;
		this.learning = learning;
		
		int weights = input[0].length * hiddenNeurons + 2 * hiddenNeurons;
		
		this.population = new double[populationSize][weights];
		this.pBest = new double[populationSize][ weights];
		this.gBest = new double[weights];
		
		
	}

	public void  generatePopulation () {
		
		for (int i = 0; i < this.population.length; i++) {
			for (int j = 0; j < this.population[0].length; j++) {
				this.population[i][j] = Math.random();
				this.pBest[i][j] = this.population[i][j];
			}
		}
	}
	
	public void calc_fitness() {
		
		int inputWeights = input[0].length * this.hiddenNeurons;
		int outWeights = this.hiddenNeurons;
		
		double[] net = new double[this.hiddenNeurons];
		double netOut = 0;

		
	}
	
	public void populationAjust () {
		for (int i = 0; i < this.population.length; i++) {
			for (int j = 0; j < this.population[0].length; j++) {
				this.population[i][j] = this.population[i][j] + 2 * Math.random() * (this.pBest[i][j] - this.population[i][j]) +
																2 * Math.random() * (this.gBest[j] - this.population[i][j]);
			}
		}
	}

}


