document.getElementById('processButton').addEventListener('click', processCSV);

function processCSV() {
    const fileInput = document.getElementById('fileInput');
    if (!fileInput.files.length) {
        alert('Please upload a CSV file first.');
        return;
    }

    const file = fileInput.files[0];
    Papa.parse(file, {
        header: true,
        complete: async function(results) {
            const data = results.data;
            const updatedData = await Promise.all(
                data.map(async (row) => {
                    try {
                        const response = await fetch('http://127.0.0.1:5000/ccet/model1GPA/predict', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({
                                CR_ATUAL: row.CR_ATUAL,
                                GPA1: row.GPA1,
                                ingresso_atual: row.ingresso_atual,
                                IsTheyBusinessperson: row.IsTheyBusinessperson,
                                Categoria: row.Categoria,
                                SEXO: row.SEXO,
                                employee_student: row.employee_student,
                                NOME_CURSO: row.NOME_CURSO,
                            }),
                        });
                        const prediction = await response.json();
                        if (prediction.prediction === 1) {
                            console.log("Likely to conclude");
                            prediction.prediction = "Likely to conclude";
                        } else if (prediction.prediction=== 0) {
                            console.log("Likely to drop out");
                            prediction.prediction = "Likely to drop out";
                        } else {
                            console.log("Unexpected response from server"); 
        
                        }
                        return { ...row, Prediction: prediction.prediction }; // Ajuste a chave 'prediction' conforme a resposta do servidor
                    } catch (error) {
                        console.error('Error during fetch:', error);
                        return { ...row, Prediction: 'Error' };
                    }
                })
            );
            generateNewCSV(updatedData);
        },
    });
}

function generateNewCSV(data) {
    const csv = Papa.unparse(data);
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);

    const downloadLink = document.getElementById('downloadLink');
    downloadLink.href = url;
    downloadLink.style.display = 'block';
    downloadLink.download = 'predicted_data.csv';
    downloadLink.textContent = 'Download New CSV';
}
