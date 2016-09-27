# Projeto-Realidade-Virtual

Projeto da disciplina de Realidade Virtual (IF755) no período de 2016.2, lecionada pela professora Verônica Teichrieb.

# Estrutura de diretórios

As configurações do projeto estão intrinsecamente relacionadas com o Visual Studio 2015, e o projeto possui a seguinte estrutura básica (obs: alguns dos seguintes diretórios são ignorados no .gitignore):

+ bin
  + x64
    + Debug
    +  Release
+ data
+ dependencies
  + Eigen_3.2.9
  + OpenCV_3.1.0
+ $(ProjectDir)
  + include
  + intermediate
  + src
+ `Nome da Solution`.sln

O projeto deve possuir as seguintes configurações (para alterá-la, clique no projeto com o botão direito em `Solution Explorer -> Properties`):

+ Sob as configurações gerais (`Configuration Properties -> General`), para todas as configurações (Debug e Release):
  + Output Directory = $(SolutionDir)bin\$(PlatformShortName)\$(Configuration)
  + Intermediate Directory = $(ProjectDir)intermediate\$(PlatformShortName)\$(Configuration)
+ Sob as configurações de `Debugging` (`Configuration Properties -> Debugging`), para todas as configurações (Debug e Release):
  + Working Directory = $(OutDir)
+ Sob as configurações gerais de `C/C++` (`Configuration Properties -> C/C++ -> General`), para todas as configurações (Debug e Release):
  + Additional Include Directories:
    + `$(ProjectDir)include;`
    + `$(SolutionDir)dependencies/Eigen_3.2.9;`
    + `$(SolutionDir)dependencies/OpenCV_3.1.0/install/include;`
+ Sob as configurações gerais de `Linker` (`Configuration Properties -> Linker -> General`), para todas as configurações (Debug e Release):
  + Additional Library Directories = `$(SolutionDir)dependencies/OpenCV_3.1.0/install/$(PlatformShortName)/lib/$(Configuration);`
+ Sob as configurações de `Input` do `Linker` (`Configuration Properties -> Linker -> Input`), selecionar as bibliotecas estáticas do OpenCV para as respectivas configurações Debug e Release (as de Debug possuem o seguinte formato: `opencv_<nome_da_funcionalidade>310d.lib`; já as de Release, `opencv_<nome_da_funcionalidade>310.lib`).
