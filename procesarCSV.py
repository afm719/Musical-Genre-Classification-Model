import csv
from io import StringIO
# Verificar si este script se esta ejecutando como el programa principal
if __name__ == "__main__":
    # Abrir el archivo CSV en modo lectura
    datos = []
    with open('data/tcc_ceds_music.csv', 'rb') as csvfile:
    # Crear un lector CSV

        contenido = csvfile.read()
        reader = csv.reader(csvfile)
        # Filtra los bytes no deseados (0x9d en este caso)
        contenido_filtrado = contenido.replace(b'\x9d', b'')
        
        # Convierte los bytes filtrados de nuevo a una cadena
        contenido_filtrado_str = contenido_filtrado.decode('latin-1')
        
        # Crea un objeto StringIO para simular un archivo de texto
        csvfile_str = StringIO(contenido_filtrado_str)
        reader = csv.reader(csvfile_str)
        for fila in reader:
            datos.append(fila)
    

    with open('data/datos.csv','w', newline='', encoding='latin-1') as csvfile2:
        #crea un CSV de salida
        writer = csv.writer(csvfile2)
        writer.writerow(['name', 'lyric', 'topic'])
        i = 0
        for fila in datos:
            if fila[1] != "artist_name":
                writer.writerow([fila[2], fila[5], fila[29]])
        