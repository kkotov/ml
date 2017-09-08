example: %: %.cc DataFrame.h Tree.h TreeTrainer.h RandomForest.h csvUtils.h
	$(CXX) -Wl,--no-as-needed -g -Wall -std=c++11 -o $@ $< -lpthread
