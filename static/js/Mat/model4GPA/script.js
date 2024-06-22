new Vue({
    el: '#app',
    data: {
        formData: {
            ingresso_atual: '',
            SEXO: '',
            IsTheyBusinessperson: '',
            Categoria: '',
            GPA1: '',
            GPA2: '',
            GPA3: '',
            GPA4: '',
            grade_programming1: '',
            grade_programming2: '',
            grade_basic_math: '',
            grade_calculus1: '',
            grade_linear_algebra: '',
            employee_student: ''
        },
        prediction: "<p>Make a prediction</P>"
    },
    methods: {
        submitForm() {
            fetch('http://127.0.0.1:5000/model4GPA/predict', { // Use o endereço correto da sua API aqui
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(this.formData)
            })
            .then(response => response.json())
            .then(data => {
                console.log(data);
                if (data.prediction === 1) {
                    console.log("Likely to conclude");
                    this.prediction = `<p style="color: green;">likely to conclude</p>`;
                } else if (data.prediction === 0) {
                    console.log("Likely to drop out");
                    this.prediction = `<p style="color: red;">likely to drop out</p>`;
                } else {
                    this.prediction = `<p>Unexpected response from server</p>`;
                    console.log("Unexpected response from server"); 

                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert("Algo deu errado");
            });
        }
    }
});

