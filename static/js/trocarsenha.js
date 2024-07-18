new Vue({
    el: '#app',
    data: {
      novaSenha: '',
      confirmaSenha: '',
      email: ''
    },
    methods: {
      trocarSenha() {
        if (this.novaSenha !== this.confirmaSenha) {
          alert("As senhas precisam ser iguais");
          return;
        }

        axios.post('/senha', {
          nova_senha: this.novaSenha,
          confirma_senha: this.confirmaSenha,
          email: this.email
        })
        .then(response => {
          if (response.data.success) {
            alert('Senha atualizada com sucesso!');
          } else {
            alert('Erro ao atualizar a senha: ' + response.data.message);
          }
        })
        .catch(error => {
          console.log(error);
          alert('Erro ao atualizar a senha.');
        });
      }
    }
  });