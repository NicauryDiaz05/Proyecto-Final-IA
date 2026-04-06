# Proyecto-de-Final-IA

## Nombre 
Nicaury Carolina Diaz Amador 

## Matrícula
 23-SISN-2-028

## Proyecto
Letrova es una aplicación inteligente que clasifica libros por género literario, tipo de lectura e identidad del autor, y traduce textos al español preservando el tono y las emociones del original. Acepta texto directo, archivos (PDF, EPUB, DOCX) y fotografías de páginas tomadas desde la cámara del dispositivo

PROPUESTA DE PROYECTO

Titulo: Sistema Inteligente de Clasificacion y Traduccion Literaria de Libros al español

DESCRIPCION GENERAL

La aplicacion permite a ulos usuarios subir o ingresar fragmentos de texto de cualquier libro o libros completos, y el sistema realiza dos funciones principales: clasificar el libro segun su genero literario, tipo de lectura y estilo de autor, y ofrecer una traduccion humanizada al español que preserve el tono y estilo literario del texto original, permitiendo que no se pierda la emociones y efoques principal del libro en su idioma original.

MODULO 1: CLASIFICACION INTELIGENTE DE LIBROS

El primer modulo se encarga de analizar el contenido textual de un libro o fragmento para extraer tres categorias clave:

<<<<<<< HEAD
El primer modulo se encarga de analizar el contenido textual de un libro o
fragmento para extraer tres categorias clave:

- Genero literario:Novela, Cuento, Poesía, Ensayo, Teatro, Fábula, Crónica
- Tipo de lectura: academico, entretenimiento, infantil, juvenil .
- Identificacion del autor: se analiza el contenido del texto para determinar
  quien es el autor de la obra.

=======
Genero literario:Novela, Cuento, Poesía, Ensayo, Teatro, Fábula, Crónica
Tipo de lectura: academico, entretenimiento, infantil, juvenil .
Identificacion del autor: se analiza el contenido del texto para determinar quien es el autor de la obra.

Se utilizara un modelo de clasificacion de texto basado en arquitecturas de lenguaje tipo BERT o similares, entrenado o ajustado (fine-tuning) sobre un conjunto de datos de libros previamente etiquetados por genero, tipo y autor.

El modelo recibe el fragmento de texto como entrada y produce como salida las categorias correspondientes junto con la identificacion del autor.

Para la identificacion del autor, el modelo analiza el contenido semantico del texto y lo contrasta contra una base de datos de obras literarias conocidas, devolviendo el nombre del autor, su nacionalidad, la epoca en que escribio y un listado de sus obras mas representativas.

Como fuente de datos se pueden utilizar colecciones de libros de dominio publico disponibles en el Proyecto Gutenberg, que ofrece miles de obras etiquetadas con informacion del autor, complementadas con APIs de informacion literaria como Open Library o Google Books API para enriquecer los datos biograficos del autor identificado.

MODULO 2: TRADUCCION HUMANIZADA AL ESPANOL

El segundo modulo se enfoca en traducir fragmentos de libros o libros completos al español de una manera que va mas alla de la traduccion literal o mecanica. El objetivo es que la traduccion se sienta natural, fluida y fiel al estilo del autor original.

Como funciona con Deep Learning / APIs:

Se utilizara una API de procesamiento de lenguaje natural con capacidades avanzadas de generacion de texto. A diferencia de los traductores automaticos convencionales que traducen palabra por palabra, este modulo funciona con un enfoque orientado al estilo literario:

Analisis previo: antes de traducir, el sistema analiza el tono del texto (poetico, dramatico, humoristico, tecnico, etc.) y las caracteristicas del autor.
 
Evaluacion de calidad: opcionalmente, se puede aplicar una metrica de evaluacion automatica (como BERTScore) para comparar la fluidez y coherencia de la traduccion generada frente a una traduccion mecanica convencional, demostrando la superioridad del enfoque humanizado.

VALOR Y APLICACION REAL

Esta aplicacion aporta valor real en los siguientes escenarios:

Lectores que desean explorar libros en otros idiomas con una traduccion de calidad literaria.
Estudiantes de literatura que necesitan clasificar obras para sus estudios.
Investigadores que trabajan con corpus de textos literarios.
Personas que quieren descubrir nuevos libros segun su estilo de lectura preferido.
Auque este proyecto suger mas de una necesidad propia de poder avanzar mis habitos licterarios sin perder la funcion principal de los libros que es trasmitir emociones y conocimiento atraves de su paginas.

La combinacion de clasificacion automatica con traduccion de calidad literaria convierte esta herramienta en una solucion unica que va mas alla de lo que ofrecen las aplicaciones de traduccion actuales.

LINK DEL VIDEO :  https://youtu.be/-UQ9TyrQRCs