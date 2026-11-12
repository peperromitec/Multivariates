def estilo1(fig):
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(
        title=dict(
            font=dict(size=24),
            x=0.5,
            xanchor="center",
            y=0.95,
            yanchor="top"
        )
    )
    return fig